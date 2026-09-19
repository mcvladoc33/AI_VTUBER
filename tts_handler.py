import os
import re
import time
import warnings
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from unicodedata import normalize

warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_BIN = os.path.join(BASE_DIR, "bin")

if FFMPEG_BIN not in os.environ["PATH"]:
    os.environ["PATH"] = FFMPEG_BIN + os.path.pathsep + os.environ["PATH"]

from pydub import AudioSegment
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from num2words import num2words

from logger_config import log
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol
import styletts2_inference.models


def fake_hf_hub_download(repo_id, filename, **kwargs):
    local_file_path = os.path.join(repo_id, filename)
    if os.path.exists(local_file_path):
        return local_file_path
    raise FileNotFoundError(f"Файл моделі не знайдено локально: {local_file_path}")


styletts2_inference.models.hf_hub_download = fake_hf_hub_download

original_open = open


def utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)


styletts2_inference.models.open = utf8_open


class TTSHandler:
    def __init__(self, config):
        self.config = config
        self.tts_config = config.get('tts', {})

        # Який рушій синтезу використовувати: "pytorch" (дефолт, з підтримкою
        # клонування голосу mode=2) або "onnx" (швидше на CPU, лише пресети)
        self.engine = str(self.tts_config.get("engine", "pytorch")).strip().lower()
        if self.engine not in ("pytorch", "onnx"):
            log.warning(f"⚠️ [TTS] Невідомий tts.engine='{self.engine}', відкочуюсь на 'pytorch'.")
            self.engine = "pytorch"

        # Обмежуємо потоки ДО створення моделі — інакше рушій хапає всі
        # доступні ядра і конкурує з LLM-decode за той самий бюджет
        tts_threads = self.tts_config.get("n_threads", 2)
        torch.set_num_threads(tts_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch дозволяє встановити interop-потоки лише ОДИН раз за
            # весь час життя процесу, незалежно від значення. У main.py
            # TTSHandler створюється рівно один раз — це ніколи не
            # спрацьовує. Але benchmark.py навмисно створює по TTSHandler
            # на кожен варіант конфігурації В ОДНОМУ процесі (щоб не
            # платити за перезапуск Python щоразу) — і другий виклик з
            # ТИМ САМИМ значенням 1 все одно кидає RuntimeError. Значення
            # від першого виклику вже діє на весь процес, тож просто
            # ігноруємо повторну спробу, а не падаємо.
            pass
        log.info(f"🧵 [TTS] Кількість потоків обмежено до {tts_threads} (рушій: {self.engine.upper()}).")

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.styletts_path = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
        self.verbalizer_path = os.path.join(BASE_DIR, "models", "mbart-large-50-verbalization")
        self.preset_dir = os.path.join(BASE_DIR, "voices")
        self.ref_dir = os.path.join(BASE_DIR, "references")
        self.output_dir = os.path.join(BASE_DIR, "outputs")

        os.makedirs(self.preset_dir, exist_ok=True)
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f"📥 [TTS] Ініціалізація StyleTTS2 UA (Робота на: {self.device.upper()})...")

        self.mode = self.tts_config.get("mode", "1")
        self.speed = self.tts_config.get("speed", 1.15)
        self.noise_scale = self.tts_config.get("noise_scale", 0.05)
        self.match_duration = self.tts_config.get("match_duration", False)
        self.use_verbalizer = self.tts_config.get("use_verbalizer", False)

        # Максимальна довжина ОДНОГО шматка для одного виклику StyleTTS2.
        # Не про мердж кількох речень докупи (той більше не робимо — див.
        # _split_to_parts), а про те, коли ОДНЕ речення настільки довге,
        # що його варто розрізати по комах/тире, щоб не змушувати
        # користувача чекати на весь монолог одним викликом.
        self.max_chunk_chars = self.tts_config.get("max_chunk_chars", 140)

        # Скільки шматків тексту синтезувати ОДНОЧАСНО замість по черзі.
        # 1 = стара послідовна поведінка. Синтез кожного шматка все одно
        # обмежений спільним пулом потоків рушія (torch.set_num_threads /
        # onnxruntime intra_op_num_threads = tts.n_threads) — тому паралель
        # тут не про "більше ядер", а про краще заповнення простоїв
        # усередині одного forward-проходу короткими шматками.
        # ОБЕРЕЖНО з engine="pytorch": конкурентна безпека самого forward
        # виклику StyleTTS2 (styletts2_inference) під питанням — бібліотека
        # не документує це явно. engine="onnx" безпечніший — ONNX Runtime
        # офіційно підтримує паралельні Run() на одній сесії.
        self.parallel_chunks = max(1, int(self.tts_config.get("parallel_chunks", 1)))
        self._synth_pool = ThreadPoolExecutor(
            max_workers=self.parallel_chunks,
            thread_name_prefix="tts-synth"
        )

        self.is_first_chunk = True
        self._session_wavs = []
        # Час завершення відтворення попереднього шматка в межах ПОТОЧНОЇ
        # репліки. За ним _audio_playback_worker визначає реальну "тишу" —
        # паузу, яку фактично чує користувач між шматками, незалежно від
        # того, LLM чи TTS її спричинили. None між репліками (див.
        # reset_session), щоб природна пауза "чекаю наступного вводу" не
        # плуталась із паузою всередині монологу.
        self._last_playback_finish = None

        # Ті самі числа, що йдуть у log.info нижче, але у "сирому" вигляді
        # для зовнішніх інструментів (напр. benchmark.py) — щоб не парсити
        # текст логів. chunk_metrics: [{"text","chars","elapsed"}, ...] —
        # один запис на кожен синтезований шматок ПОТОЧНОЇ репліки.
        # silence_log: [float, ...] — КОЖНА виміряна пауза перед шматком,
        # включно з тими, що нижче порогу 0.3с для друку в лог.
        self.chunk_metrics = []
        self.silence_log = []

        # УВАГА: це окремий токенайзер mBART для вербалізації чисел, а НЕ
        # токенайзер StyleTTS2 (той — self.tokenizer, див. нижче). Раніше
        # обидва звались self.tokenizer — коли додався перемикач рушіїв,
        # ініціалізація рушія переписувала цей об'єкт значенням токенайзера
        # StyleTTS2 вже ПІСЛЯ цього блоку, тихо ламаючи use_verbalizer=true
        # (self.tokenizer(text) викликав би не той токенайзер). Розведено
        # по різних іменах, щоб цей клас конфліктів більше не повторювався.
        self.verbalizer_model = None
        self.verbalizer_tokenizer = None
        if self.use_verbalizer:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            try:
                self.verbalizer_tokenizer = MBart50TokenizerFast.from_pretrained(
                    self.verbalizer_path, local_files_only=True
                )
                self.verbalizer_model = MBartForConditionalGeneration.from_pretrained(
                    self.verbalizer_path, local_files_only=True
                ).to(self.device)
                self.verbalizer_tokenizer.src_lang = "uk_UA"
                self.verbalizer_tokenizer.tgt_lang = "uk_UA"
                log.info("✅ [TTS] Вербалізатор mBART успішно завантажено.")
            except Exception as e:
                log.warning(f"⚠️ [TTS] mBART не завантажено ({e}), працює алгоритмічна заміна.")

        # self.multi_model лишається None для ONNX-рушія — весь стан живе
        # в self.onnx_session, а self.tokenizer (StyleTTS2, не mBART вище)
        # спільний для обох гілок
        self.multi_model = None
        self.onnx_session = None
        self.tokenizer = None

        if self.engine == "onnx":
            self._init_onnx_engine()
        else:
            self._init_pytorch_engine()

        self.stressify = Stressifier()
        self.ipa_func = ipa

        self.style = None
        self.target_duration = None
        self._prepare_voice()

        try:
            warm = normalize('NFKC', "Привіт")
            ps = self.ipa_func(self.stressify(warm))
            if ps:
                tokens = self.tokenizer.encode(ps)
                warm_style = self._clone_style(self.style)
                _ = self._synthesize(tokens, self.speed, warm_style)
            log.info("✅ [TTS] Прогрів синтезатора завершено.")
        except Exception as e:
            log.warning(f"⚠️ [TTS] Прогрів TTS не вдався: {e}")

        threading.Thread(target=self._text_processing_worker, daemon=True).start()
        threading.Thread(target=self._audio_playback_worker, daemon=True).start()

    def _init_pytorch_engine(self):
        from styletts2_inference.models import StyleTTS2
        self.multi_model = StyleTTS2(hf_path=self.styletts_path, device=self.device)
        self.tokenizer = self.multi_model.tokenizer

        try:
            if hasattr(self.multi_model, 'model'):
                self.multi_model.model.diffusion_steps = 1
                if hasattr(self.multi_model.model, 'args'):
                    self.multi_model.model.args.diffusion_steps = 1
        except Exception:
            pass

    def _init_onnx_engine(self):
        # Лінивий імпорт — onnxruntime не потрібен, якщо engine="pytorch",
        # тож не змушуємо всіх ставити зайву залежність
        import onnxruntime as ort
        from styletts2_inference.models import StyleTTS2Tokenizer

        onnx_path = self.tts_config.get("onnx_model_path", "models/styletts2.onnx")
        if not os.path.isabs(onnx_path):
            onnx_path = os.path.join(BASE_DIR, onnx_path)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"❌ [TTS] ONNX-модель не знайдено: {onnx_path}")

        tts_threads = self.tts_config.get("n_threads", 2)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = tts_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.onnx_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        # Токенайзер локальний, hf_hub_download вже запатчено вище на офлайн-шлях
        self.tokenizer = StyleTTS2Tokenizer(hf_path=self.styletts_path)

        log.info(f"⚡ [TTS] ONNX Runtime ініціалізовано ({os.path.basename(onnx_path)}, "
                 f"{tts_threads} потоків).")

    def _clone_style(self, style):
        """Копія вектора стилю, сумісна з поточним рушієм (torch.Tensor або np.ndarray)."""
        if self.engine == "onnx":
            return style.copy()
        return style.clone()

    def _add_style_noise(self, style):
        """Додає невеликий шум до стилю для природної варіативності голосу.

        Для ONNX-гілки навмисно НЕ використовує np.random.randn/np.random.*
        (глобальний RNG NumPy не є потокобезпечним) — при паралельному
        синтезі (tts.parallel_chunks > 1) кілька потоків одночасно писали б
        у той самий глобальний стан. np.random.default_rng() створює
        незалежний генератор на кожен виклик, тож гонки даних немає."""
        if self.noise_scale <= 0:
            return style
        if self.engine == "onnx":
            rng = np.random.default_rng()
            noise = rng.standard_normal(style.shape).astype(np.float32) * self.noise_scale
            return style + noise
        style += torch.randn_like(style) * self.noise_scale
        return style

    def _synthesize(self, tokens, speed, style):
        """Єдина точка синтезу — повертає float32 numpy-хвилю незалежно від рушія."""
        if self.engine == "onnx":
            tokens_np = tokens.numpy().astype(np.int64) if hasattr(tokens, "numpy") else np.asarray(tokens, dtype=np.int64)
            style_np = style if isinstance(style, np.ndarray) else np.asarray(style, dtype=np.float32)
            inputs = {
                "tokens": tokens_np,
                "speed": np.array(speed, dtype=np.float32),
                "s_prev": style_np.astype(np.float32),
            }
            wav = self.onnx_session.run(None, inputs)[0]
            return np.asarray(wav).flatten()

        wav = self.multi_model(tokens, speed=speed, s_prev=style)
        return wav.cpu().numpy().flatten()

    def _synthesize_part(self, t, speed):
        """Синтезує ОДНУ вже нарізану частину тексту. Винесено окремо від
        _text_processing_worker, щоб можна було виконувати кілька частин
        одночасно через self._synth_pool (tts.parallel_chunks) — поки одна
        ще рахується, наступна вже стартує, а не чекає своєї черги.
        Повертає (audio_chunk, elapsed_seconds), audio_chunk=None якщо
        після очищення тексту не лишилось фонем для синтезу."""
        part_start = time.time()
        t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
        ps = self.ipa_func(self.stressify(t_norm))
        if not ps:
            return None, 0.0

        tokens = self.tokenizer.encode(ps)

        if self.style is None:
            raise ValueError("Об'єкт стилю не ініціалізовано.")

        current_style = self._clone_style(self.style)
        current_style = self._add_style_noise(current_style)

        audio_chunk = self._synthesize(tokens, speed, current_style)
        return audio_chunk, time.time() - part_start

    def _prepare_voice(self):
        mode_str = str(self.mode).strip()
        if mode_str == "2":
            # Динамічне клонування голосу потребує voice-encoder з повної
            # PyTorch-моделі — ONNX-граф його не містить (лише синтез за
            # готовим вектором стилю), тож ця комбінація не підтримується.
            if self.engine == "onnx":
                raise ValueError(
                    "❌ [TTS] Режим клонування голосу (tts.mode=2) не підтримується разом з "
                    "tts.engine=onnx. Використай mode=1 з готовим пресетом, або engine=pytorch."
                )

            ref_file = self.tts_config.get("reference_filename", "sample.wav")
            ref_path = os.path.join(self.ref_dir, ref_file)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"❌ Референс '{ref_file}' не знайдено.")

            self.style = self.multi_model.extract_voice_features(ref_path)
            if isinstance(self.style, list):
                self.style = self.style[-1]
            self.style = self.style.to(self.device)

            y, _ = librosa.load(ref_path, sr=24000)
            self.target_duration = librosa.get_duration(y=y, sr=24000)
            log.info(f"🎭 [TTS] Клонування голосу з файлу: {ref_file}")
        else:
            preset_file = self.tts_config.get("preset_filename", "Інна Гелевера.pt")
            preset_path = os.path.join(self.preset_dir, preset_file)
            if not os.path.exists(preset_path):
                raise FileNotFoundError(f"❌ Пресет '{preset_file}' не знайдено.")

            raw_style = torch.load(preset_path, map_location='cpu')

            if self.engine == "onnx":
                # ONNX-сесія працює з float32 numpy-масивами, не з torch.Tensor
                arr = raw_style.detach().cpu().numpy().astype(np.float32)
                if arr.ndim == 1:
                    arr = np.expand_dims(arr, axis=0)
                self.style = arr
            else:
                self.style = raw_style.to(self.device)

            log.info(f"👤 [TTS] Успішно активовано пресет: {preset_file}")

    def _split_to_parts(self, text_data):
        # Ключова зміна порівняно зі старою версією: тут БІЛЬШЕ НЕ мерджимо
        # кілька завершених речень в один блок до MAX_CHARS. Раніше блок
        # "Мені потрібні деталі, друже. Ти ж не хочеш нудної відповіді від
        # Селті. Я готова до епічних описів." (98 симв., 3 речення) йшов
        # ОДНИМ викликом StyleTTS2 і користувач чекав 9+ секунд до першого
        # звуку. Тепер кожне завершене речення — окремий виклик: перше
        # аудіо звучить за час одного речення (~3-4с), а не всього блоку.
        # ВАЖЛИВО: тут лишились ЛИШЕ справжні кінці речення (. ! ? …).
        # Тире (—/–), двокрапка й крапка з комою — НЕ кінці речення,
        # вони йдуть у _split_long_sentence нижче. Раніше вони теж
        # ловились тут, і речення типу "Чотири — це вже не так просто"
        # розривалось прямо на тире як на дві "сентенції" — звідси й
        # ефект заїкання/обрубаних фраз у TTS.
        sentences = re.split(r'([.!?…])(?=(?:[^"]*"[^"]*")*[^"]*$)(?=(?:[^«]*«[^»]*»)*[^»]*$)', text_data)
        raw, cur = [], ""
        for item in sentences:
            if not item:
                continue
            if item in '.!?…':
                cur += item
                raw.append(cur.strip())
                cur = ""
            else:
                cur += item
        if cur.strip():
            raw.append(cur.strip())

        # Єдиний виняток — геть крихітні "хвостики" (типу самотнього "Так."
        # після попереднього речення): їх усе ж клеїмо до сусіда, інакше
        # плодимо виклики StyleTTS2 з фіксованим оверхедом ~4с на секунди
        # користі. MIN_STANDALONE навмисно малий — це не про мердж речень
        # заради швидкості, а про відсікання виродкових уламків парсингу.
        MIN_STANDALONE = 12
        merged = []
        for s in raw:
            if not s:
                continue
            if merged and len(s) < MIN_STANDALONE:
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)

        final = []
        for chunk in merged:
            if len(chunk) <= self.max_chunk_chars:
                final.append(chunk)
                continue
            final.extend(self._split_long_sentence(chunk))

        return [p for p in final if p]

    def _split_long_sentence(self, sentence):
        """Ріже ОДНЕ задовге речення по логічних розділових знаках (кома,
        крапка з комою, тире) — це звучить природніше за розрив по слову
        і дає той самий ефект швидшого першого звуку для речень, які самі
        по собі довші за self.max_chunk_chars."""
        pieces = re.split(r'([,;:—–])', sentence)
        parts, cur = [], ""
        for item in pieces:
            if not item:
                continue
            if item in ',;:—–':
                cur += item
                parts.append(cur.strip())
                cur = ""
            else:
                cur += item
        if cur.strip():
            parts.append(cur.strip())

        # Клеїмо сусідні шматки коми, поки влазять у ліміт — щоб не
        # озвучувати кожну кому окремим мікро-викликом
        merged = []
        acc = ""
        for p in parts:
            if not p:
                continue
            if not acc:
                acc = p
            elif len(acc) + len(p) + 1 <= self.max_chunk_chars:
                acc += " " + p
            else:
                merged.append(acc)
                acc = p
        if acc:
            merged.append(acc)

        # Фолбек: якщо в реченні взагалі немає ком/тире (тому не розрізалось
        # вище) і воно все ще задовге — ріжемо по словах, як і раніше
        final = []
        for chunk in merged:
            if len(chunk) <= self.max_chunk_chars:
                final.append(chunk)
                continue
            words, temp = chunk.split(' '), ""
            for w in words:
                if len(temp) + len(w) + 1 > self.max_chunk_chars and temp:
                    final.append(temp.strip())
                    temp = w
                else:
                    temp += " " + w if temp else w
            if temp.strip():
                final.append(temp.strip())

        return final

    def _text_processing_worker(self):
        while True:
            text = self.text_queue.get()
            if text is None:
                continue

            if not text.strip():
                self.text_queue.task_done()
                continue

            try:
                clean_text = text.strip()
                clean_text = re.sub(r'\.{2,}', '.', clean_text)
                clean_text = re.sub(r'[^\w\s\d.,!?;:()\-—–\'"\«\»]', '', clean_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                if not clean_text:
                    self.text_queue.task_done()
                    continue

                raw_sentences = re.split(r'(?<=[.!?;:])\s+', clean_text)
                processed_sentences = []

                for s in raw_sentences:
                    if not s.strip():
                        continue
                    if self.use_verbalizer and self.verbalizer_model and re.search(r'\d+', s):
                        inputs = self.verbalizer_tokenizer(s, return_tensors="pt", padding=True).to(self.device)
                        generated_tokens = self.verbalizer_model.generate(
                            **inputs,
                            forced_bos_token_id=self.verbalizer_tokenizer.lang_code_to_id["uk_UA"],
                            max_length=len(s) + 40,
                            no_repeat_ngram_size=3,
                            early_stopping=True
                        )
                        clean_s = self.verbalizer_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                        processed_sentences.append(clean_s.strip())
                    else:
                        processed_sentences.append(s.strip())

                final_text = " ".join(processed_sentences)
                final_text = re.sub(r'\badvel\b', 'Адвел', final_text, flags=re.IGNORECASE)

                if re.search(r'\d+', final_text):
                    final_text = re.sub(r'\d+', lambda m: num2words(int(m.group(0)), lang='uk'), final_text)

                parts = self._split_to_parts(final_text)
                final_speed = self.speed

                mode_str = str(self.mode).strip()
                if mode_str == "2" and self.match_duration and self.target_duration:
                    # mode=2 гарантовано означає engine=pytorch (перевірено в _prepare_voice)
                    temp_wavs = []
                    for t in parts:
                        t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
                        ps = self.ipa_func(self.stressify(t_norm))
                        if ps:
                            tokens = self.tokenizer.encode(ps)
                            w = self._synthesize(tokens, 1.0, self.style)
                            temp_wavs.append(w)
                    if temp_wavs:
                        gen_len = sum(len(w) for w in temp_wavs) / 24000
                        calc_speed = gen_len / self.target_duration
                        if 0.6 <= calc_speed <= 1.4:
                            final_speed = calc_speed

                block_wavs = []

                # Всі частини блоку відправляються в пул одразу (submit),
                # а результати забираються В ПОРЯДКУ НАДХОДЖЕННЯ через
                # future.result() — навіть якщо шматок 3 порахувався
                # раніше за шматок 2, у чергу відтворення він потрапить
                # лише після 2-го. Паралелізм є (при parallel_chunks > 1
                # кілька .result() вже виконуються у фоні одночасно), а
                # порядок мовлення лишається природним.
                futures = [self._synth_pool.submit(self._synthesize_part, t, final_speed) for t in parts]

                for i, (t, fut) in enumerate(zip(parts, futures), 1):
                    audio_chunk, part_time = fut.result()
                    if audio_chunk is None:
                        continue

                    block_wavs.append(audio_chunk)

                    self.chunk_metrics.append({"text": t, "chars": len(t), "elapsed": part_time})

                    log.info(
                        f"   📢 [StyleTTS2] Шматок {i}/{len(parts)} готовий за {part_time:.3f}s! ({len(t)} симв.) -> {t}")

                    self.audio_queue.put(audio_chunk)

                if block_wavs:
                    self._session_wavs.extend(block_wavs)

            except Exception as e:
                log.error(f"❌ ПОМИЛКА [TTS]: {e}")

            self.text_queue.task_done()

    def flush_session_audio(self):
        """Кодує монолог один раз, коли CPU вже вільний — не під час генерації."""
        if not self._session_wavs:
            return
        try:
            combined = np.concatenate(self._session_wavs)
            sf.write(os.path.join(self.output_dir, "output.wav"), combined, 24000)
            int16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
            AudioSegment(int16.tobytes(), frame_rate=24000, sample_width=2, channels=1) \
                .export(os.path.join(self.output_dir, "output.mp3"), format="mp3", bitrate="192k")
            log.info("   💾 [SYSTEM] Монолог збережено в output.mp3")
        except Exception as e:
            log.warning(f"⚠️ [SYSTEM] Не вдалося зберегти аудіо: {e}")
        finally:
            self._session_wavs = []

    def _audio_playback_worker(self):
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None:
                continue

            if self._last_playback_finish is not None:
                silence = time.time() - self._last_playback_finish
                self.silence_log.append(silence)
                # Поріг 0.3с — щоб не спамити на дрібних, неминучих затримках
                # планувальника ОС; це саме та "тиша", яку реально чує
                # користувач між двома шматками ОДНІЄЇ репліки. Поріг лише
                # для друку в лог — у silence_log вище пишемо БУДЬ-яке
                # значення, щоб benchmark.py мав повну картину.
                if silence > 0.3:
                    log.info(f"   🤫 [SYSTEM] Тиша {silence:.2f}s перед наступним шматком")

            try:
                sd.play(audio_data, 24000)
                sd.wait()
            except Exception as play_err:
                log.warning(f"⚠️ [Playback Error]: {play_err}")
            finally:
                self.audio_queue.task_done()
                self._last_playback_finish = time.time()

    def play_text_async(self, text: str):
        if not text.strip():
            return
        self.text_queue.put(text)

    def wait_until_done(self):
        self.text_queue.join()
        self.audio_queue.join()

    def generate_speech(self, text: str):
        self.play_text_async(text)

    def reset_session(self):
        self.is_first_chunk = True
        self._session_wavs = []
        # Скидаємо ДО початку нової репліки — природна пауза "чекаю на
        # користувача" між репліками не повинна рахуватись як "тиша"
        self._last_playback_finish = None
        self.chunk_metrics = []
        self.silence_log = []

    def shutdown(self):
        """Акуратне завершення пулу потоків синтезу. main.py живе один раз
        за весь процес і в цьому не потребує — але інструменти на кшталт
        benchmark.py створюють по TTSHandler на кожну тестовану
        конфігурацію в одному процесі, і без явного shutdown пули потоків
        від попередніх варіантів накопичувались би до кінця скрипта."""
        try:
            self._synth_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass