import os
import re
import time
import warnings
from unicodedata import normalize

# Ігноруємо попередження бібліотек
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Вимикаємо офлайн-попередження HuggingFace
# os.environ["BITSANDBYTES_NOWELCOME"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch

# Спроба підключити відеокарту Intel, якщо вона є
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from num2words import num2words

from logger_config import log
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol
import styletts2_inference.models

# --- Заглушки для локального завантаження моделей ---
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
# ---------------------------------------------------

class TTSHandler:
    def __init__(self, config):
        self.config = config
        self.tts_config = config.get('tts', {})

        # Обмежуємо потоки згідно з конфігом
        tts_threads = self.tts_config.get("n_threads", 3)
        torch.set_num_threads(tts_threads)
        torch.set_num_interop_threads(1)
        log.info(f"🧵 [TTS] Кількість потоків PyTorch обмежено до {tts_threads}.")

        # Шляхи до файлів
        self.styletts_path = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
        self.verbalizer_path = os.path.join(BASE_DIR, "models", "mbart-large-50-verbalization")
        self.preset_dir = os.path.join(BASE_DIR, "voices")
        self.ref_dir = os.path.join(BASE_DIR, "references")
        self.output_dir = os.path.join(BASE_DIR, "outputs")

        os.makedirs(self.preset_dir, exist_ok=True)
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Вибір пристрою
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            self.device = 'xpu'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        log.info(f"📥 [TTS] Ініціалізація StyleTTS2 UA (Робота на: {self.device.upper()})...")

        # Налаштування
        self.mode = str(self.tts_config.get("mode", "1")).strip()
        self.speed = self.tts_config.get("speed", 1.15)
        self.noise_scale = self.tts_config.get("noise_scale", 0.05)
        self.use_verbalizer = self.config.get("memory", {}).get("use_verbalizer", False)

        # Буфер для збереження аудіо поточної сесії
        self._session_wavs = [] 

        self._load_model()
        self.stressify = Stressifier()
        self.ipa_func = ipa
        
        self.style = None
        self._prepare_voice()

    def _load_model(self):
        """Внутрішній метод для завантаження моделей."""
        self.verbalizer_model = None
        self.tokenizer = None
        
        # Вербалізатор (за замовчуванням вимкнений для економії RAM)
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

        # Основна модель StyleTTS2
        from styletts2_inference.models import StyleTTS2
        self.multi_model = StyleTTS2(hf_path=self.styletts_path, device=self.device)

        try:
            if hasattr(self.multi_model, 'model'):
                self.multi_model.model.diffusion_steps = 1
                if hasattr(self.multi_model.model, 'args'):
                    self.multi_model.model.args.diffusion_steps = 1
        except Exception:
            pass

    def _prepare_voice(self):
        """Внутрішній метод для завантаження голосу (пресету або клону)."""
        if self.mode == "2":
            ref_file = self.tts_config.get("reference_filename", "sample.wav")
            ref_path = os.path.join(self.ref_dir, ref_file)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"❌ Референс '{ref_file}' не знайдено.")

            self.style = self.multi_model.extract_voice_features(ref_path)
            if isinstance(self.style, list):
                self.style = self.style[-1]
            self.style = self.style.to(self.device)
            log.info(f"🎭 [TTS] Клонування голосу з файлу: {ref_file}")
        else:
            preset_file = self.tts_config.get("preset_filename", "Інна Гелевера.pt")
            preset_path = os.path.join(self.preset_dir, preset_file)
            if not os.path.exists(preset_path):
                raise FileNotFoundError(f"❌ Пресет '{preset_file}' не знайдено.")

            self.style = torch.load(preset_path, map_location=self.device)
            log.info(f"👤 [TTS] Успішно активовано пресет: {preset_file}")

    def warmup(self) -> None:
        """Метод для 'прогріву' моделі перед першим використанням."""
        try:
            log.info("⏳ [TTS] Починаємо прогрів синтезатора...")
            warm_text = self._normalize_text("Привіт")
            _ = self.synthesize_chunk(warm_text)
            log.info("✅ [TTS] Прогрів синтезатора завершено.")
        except Exception as e:
            log.warning(f"⚠️ [TTS] Прогрів TTS не вдався: {e}")

    def _normalize_text(self, text: str) -> str:
        """Очищення тексту та заміна чисел на слова."""
        clean_text = text.strip()
        clean_text = re.sub(r'\.{2,}', '.', clean_text)
        clean_text = re.sub(r'[^\w\s\d.,!?;:()\-—–\'"\«\»]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = re.sub(r'\badvel\b', 'Адвел', clean_text, flags=re.IGNORECASE)

        if re.search(r'\d+', clean_text):
            clean_text = re.sub(r'\d+', lambda m: num2words(int(m.group(0)), lang='uk'), clean_text)
            
        return clean_text

    def start_session(self) -> None:
        """Починає нову сесію (репліку). Очищує буфер аудіо."""
        self._session_wavs = []

    def _synthesize_raw(self, text: str):
        """Перетворює текст у масив звукових даних (numpy)."""
        t_norm = normalize('NFKC', text.replace('+', StressSymbol.CombiningAcuteAccent))
        ps = self.ipa_func(self.stressify(t_norm))
        
        if not ps:
            return None
            
        tokens = self.multi_model.tokenizer.encode(ps)
        current_style = self.style.clone().to(self.device)
        
        if self.noise_scale > 0:
            current_style += torch.randn_like(current_style) * self.noise_scale

        wav = self.multi_model(tokens, speed=self.speed, s_prev=current_style)
        return wav.cpu().numpy().flatten()

    def synthesize_chunk(self, text: str):
        """Нормалізує та синтезує шматок тексту."""
        clean_text = self._normalize_text(text)
        if not clean_text:
            return None
            
        start_time = time.time()
        audio_data = self._synthesize_raw(clean_text)
        elapsed = time.time() - start_time
        
        if audio_data is not None:
            log.info(f"   📢 [StyleTTS2] Синтезовано за {elapsed:.3f}s! ({len(clean_text)} симв.) -> {clean_text}")
            
        return audio_data

    def play_chunk(self, audio) -> None:
        """Відтворює аудіо через системні динаміки та чекає завершення."""
        if audio is None:
            return
            
        try:
            sd.play(audio, 24000)
            sd.wait() # Програма зупиняється тут, поки шматок не дограє (послідовна робота)
        except Exception as play_err:
            log.warning(f"⚠️ [Playback Error]: {play_err}")

    def synthesize_and_play(self, text: str) -> None:
        """Об'єднує синтез, відтворення та збереження в буфер для одного шматка."""
        audio_data = self.synthesize_chunk(text)
        if audio_data is not None:
            self._session_wavs.append(audio_data)
            self.play_chunk(audio_data)

    def finish_session(self) -> None:
        """Завершує сесію. Викликається після того, як всі шматки озвучені."""
        self.flush_session_audio()

    def flush_session_audio(self) -> str | None:
        """Зберігає всю наговорену репліку у WAV файл. Без ffmpeg."""
        if not self._session_wavs:
            return None
            
        try:
            combined = np.concatenate(self._session_wavs)
            out_path = os.path.join(self.output_dir, "output.wav")
            sf.write(out_path, combined, 24000)
            log.info(f"   💾 [SYSTEM] Репліку збережено у {out_path}")
            return out_path
        except Exception as e:
            log.warning(f"⚠️ [SYSTEM] Не вдалося зберегти аудіо: {e}")
            return None
        finally:
            self._session_wavs = []

    def close(self) -> None:
        """Звільняє ресурси при вимкненні програми."""
        log.info("🛑 [TTS] Модуль TTS успішно закрито.")
        self._session_wavs = []