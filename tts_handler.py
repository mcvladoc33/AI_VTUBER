import os
import re
import time
import warnings
import threading
import queue
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

        # Обмежуємо потоки PyTorch ДО створення моделі — інакше StyleTTS2
        # хапає всі доступні ядра і конкурує з LLM-decode за той самий бюджет
        tts_threads = self.tts_config.get("n_threads", 2)
        torch.set_num_threads(tts_threads)
        torch.set_num_interop_threads(1)
        log.info(f"🧵 [TTS] Кількість потоків PyTorch обмежено до {tts_threads}.")

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

        self.is_first_chunk = True
        self._session_wavs = []

        self.verbalizer_model = None
        self.tokenizer = None
        if self.use_verbalizer:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            try:
                self.tokenizer = MBart50TokenizerFast.from_pretrained(self.verbalizer_path, local_files_only=True)
                self.verbalizer_model = MBartForConditionalGeneration.from_pretrained(
                    self.verbalizer_path, local_files_only=True
                ).to(self.device)
                self.tokenizer.src_lang = "uk_UA"
                self.tokenizer.tgt_lang = "uk_UA"
                log.info("✅ [TTS] Вербалізатор mBART успішно завантажено.")
            except Exception as e:
                log.warning(f"⚠️ [TTS] mBART не завантажено ({e}), працює алгоритмічна заміна.")

        from styletts2_inference.models import StyleTTS2
        self.multi_model = StyleTTS2(hf_path=self.styletts_path, device=self.device)

        try:
            if hasattr(self.multi_model, 'model'):
                self.multi_model.model.diffusion_steps = 1
                if hasattr(self.multi_model.model, 'args'):
                    self.multi_model.model.args.diffusion_steps = 1
        except Exception:
            pass

        self.stressify = Stressifier()
        self.ipa_func = ipa

        self.style = None
        self.target_duration = None
        self._prepare_voice()

        try:
            warm = normalize('NFKC', "Привіт")
            ps = self.ipa_func(self.stressify(warm))
            if ps:
                _ = self.multi_model(self.multi_model.tokenizer.encode(ps),
                                      speed=self.speed, s_prev=self.style.clone())
            log.info("✅ [TTS] Прогрів синтезатора завершено.")
        except Exception as e:
            log.warning(f"⚠️ [TTS] Прогрів TTS не вдався: {e}")

        threading.Thread(target=self._text_processing_worker, daemon=True).start()
        threading.Thread(target=self._audio_playback_worker, daemon=True).start()

    def _prepare_voice(self):
        mode_str = str(self.mode).strip()
        if mode_str == "2":
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

            self.style = torch.load(preset_path, map_location=self.device)
            log.info(f"👤 [TTS] Успішно активовано пресет: {preset_file}")

    def _split_to_parts(self, text_data):
        MAX_CHARS = 240

        sentences = re.split(r'([.!?;:—–])(?=(?:[^"]*"[^"]*")*[^"]*$)(?=(?:[^«]*«[^»]*»)*[^»]*$)', text_data)
        raw, cur = [], ""
        for item in sentences:
            if not item:
                continue
            if item in '.!?;:—–':
                cur += item
                raw.append(cur.strip())
                cur = ""
            else:
                cur += item
        if cur.strip():
            raw.append(cur.strip())

        merged = []
        acc = ""
        for s in raw:
            if not s:
                continue
            if not acc:
                acc = s
            elif len(acc) + len(s) + 1 <= MAX_CHARS:
                acc += " " + s
            else:
                merged.append(acc)
                acc = s
        if acc:
            merged.append(acc)

        final = []
        for chunk in merged:
            if len(chunk) <= MAX_CHARS:
                final.append(chunk)
                continue
            words, temp = chunk.split(' '), ""
            for w in words:
                if len(temp) + len(w) + 1 > MAX_CHARS and temp:
                    final.append(temp.strip())
                    temp = w
                else:
                    temp += " " + w if temp else w
            if temp.strip():
                final.append(temp.strip())

        return [p for p in final if p]

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
                        inputs = self.tokenizer(s, return_tensors="pt", padding=True).to(self.device)
                        generated_tokens = self.verbalizer_model.generate(
                            **inputs,
                            forced_bos_token_id=self.tokenizer.lang_code_to_id["uk_UA"],
                            max_length=len(s) + 40,
                            no_repeat_ngram_size=3,
                            early_stopping=True
                        )
                        clean_s = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
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
                    temp_wavs = []
                    for t in parts:
                        t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
                        ps = self.ipa_func(self.stressify(t_norm))
                        if ps:
                            tokens = self.multi_model.tokenizer.encode(ps)
                            w = self.multi_model(tokens, speed=1.0, s_prev=self.style)
                            temp_wavs.append(w)
                    if temp_wavs:
                        gen_len = sum(len(w) for w in temp_wavs) / 24000
                        calc_speed = gen_len / self.target_duration
                        if 0.6 <= calc_speed <= 1.4:
                            final_speed = calc_speed

                block_wavs = []

                for i, t in enumerate(parts, 1):
                    part_start = time.time()
                    t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
                    ps = self.ipa_func(self.stressify(t_norm))
                    if ps:
                        tokens = self.multi_model.tokenizer.encode(ps)

                        if self.style is None:
                            raise ValueError("Об'єкт стилю не ініціалізовано.")

                        current_style = self.style.clone()
                        if self.noise_scale > 0:
                            current_style += torch.randn_like(current_style) * self.noise_scale

                        wav = self.multi_model(tokens, speed=final_speed, s_prev=current_style)
                        audio_chunk = wav.cpu().numpy().flatten()

                        block_wavs.append(audio_chunk)

                        part_time = time.time() - part_start
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
            try:
                sd.play(audio_data, 24000)
                sd.wait()
            except Exception as play_err:
                log.warning(f"⚠️ [Playback Error]: {play_err}")
            finally:
                self.audio_queue.task_done()

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