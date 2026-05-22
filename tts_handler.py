import os
import torch
import numpy as np
import sounddevice as sd
import time
from unicodedata import normalize
from logger_config import log
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier
from tts_engines import ONNXEngine
from styletts2_inference.models import StyleTTS2Tokenizer


class TTSHandler:
    def __init__(self, config):
        self.tts_config = config.get('tts', {})
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Динамічно беремо кількість потоків із конфігу (за замовчуванням 4)
        n_threads = self.tts_config.get("n_threads", 4)
        onnx_full_path = os.path.join(os.getcwd(), self.tts_config.get("onnx_path", "models/styletts2.onnx"))

        # Ініціалізуємо наш новий прискорений C++ двигун
        self.engine = ONNXEngine(onnx_full_path, n_threads=n_threads)

        self.tokenizer = StyleTTS2Tokenizer(
            hf_path=os.path.join(os.getcwd(), "models", "styletts2_ukrainian_multispeaker"))
        self.stressify = Stressifier()
        self.style = None
        self._interrupted = False

        self._load_preset()

    def _load_preset(self):
        path = os.path.join("voices", self.tts_config.get("preset_filename", "Інна Гелевера.pt"))
        if os.path.exists(path):
            # Завантажуємо ваги голосу
            loaded_style = torch.load(path, map_location='cpu')

            # Для ONNX вигідніше відразу перевести тензор у чистий NumPy масив,
            # щоб не витрачати час на конвертацію під час кожної швидкої репліки
            if hasattr(loaded_style, 'detach'):
                self.style = loaded_style.detach().cpu().numpy()
            else:
                self.style = loaded_style

            log.info(f"✅ [TTS] Пресет голосу успішно завантажено та конвертовано для ONNX: {path}")
        else:
            log.warning(f"⚠️ [TTS] Файл пресету голосу не знайдено за шляхом: {path}")

    def say(self, text):
        """Синхронний генератор і програвач фрази без внутрішніх фонових потоків бібліотеки"""
        if self._interrupted: return
        try:
            start_time = time.time()

            # 1. Нормалізація та наголоси
            cleaned_text = normalize('NFKC', text)
            stressed = self.stressify(cleaned_text)
            phonemes = ipa(stressed)
            tokens = self.tokenizer.encode(phonemes)

            # Беремо швидкість із файлу конфігурації (у тебе там 1.15)
            speed = self.tts_config.get("speed", 1.15)

            # 2. Генерація звукової хвилі через оптимізований ONNXEngine
            wav = self.engine.generate(tokens, speed, self.style)

            if wav is not None and not self._interrupted:
                audio_data = wav.astype(np.float32)
                generation_time = time.time() - start_time
                log.info(f"    🔊 [TTS] Чистий синтез завершено за: {generation_time:.2f}s")

                # 3. Пряме відтворення звуку. Потік заблокується точно на час звучання фраз, без накладок.
                sd.play(audio_data, samplerate=24000)
                sd.wait()
            elif wav is None:
                log.error("❌ [TTS] Двигун повернув пустий результат.")
        except Exception as e:
            log.error(f"❌ Помилка процесу TTS: {e}")

    def stop(self):
        self._interrupted = True
        sd.stop()
        self._interrupted = False