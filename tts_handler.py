import os
import sys
import warnings
import torch
import numpy as np
import sounddevice as sd
import threading
import queue
import time
from unicodedata import normalize
from logger_config import log
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol
import styletts2_inference.models
from tts_engines import TorchEngine, ONNXEngine
from huggingface_hub import hf_hub_download as original_hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ПАТЧІ ---
import builtins

original_open = builtins.open


def robust_utf8_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'b' not in mode and 'encoding' not in kwargs: kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)


builtins.open = robust_utf8_open
styletts2_inference.models.open = robust_utf8_open


def fake_hf_hub_download(repo_id, filename, **kwargs):
    if os.path.exists(repo_id): return os.path.join(repo_id, filename)
    return original_hf_hub_download(repo_id, filename, **kwargs)


styletts2_inference.models.hf_hub_download = fake_hf_hub_download


class TTSHandler:
    def __init__(self, config):
        self.config = config
        self.tts_config = config.get('tts', {})
        self.engine_type = self.tts_config.get("engine", "torch")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.styletts_path = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
        self.preset_dir = os.path.join(BASE_DIR, "voices")
        self.speed = self.tts_config.get("speed", 1.15)

        if self.engine_type == "onnx":
            log.info(f"🚀 [TTS] Запуск двигуна ONNX: {self.tts_config.get('onnx_path')}")
            self.engine = ONNXEngine(os.path.join(BASE_DIR, self.tts_config["onnx_path"]),
                                     self.tts_config.get("n_threads", 4))
            from styletts2_inference.models import StyleTTS2Tokenizer
            self.tokenizer = StyleTTS2Tokenizer(hf_path=self.styletts_path)
        else:
            log.info("🚀 [TTS] Запуск двигуна PyTorch")
            from styletts2_inference.models import StyleTTS2
            self.multi_model = StyleTTS2(hf_path=self.styletts_path, device=self.device)
            self.engine = TorchEngine(self.multi_model)
            self.tokenizer = self.multi_model.tokenizer

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.interrupted = False
        self._is_playing_now = False

        self.stressify = Stressifier()
        self.style = None
        self._load_preset()

        threading.Thread(target=self._text_processing_worker, daemon=True).start()
        threading.Thread(target=self._audio_playback_worker, daemon=True).start()

    def _load_preset(self):
        preset_file = self.tts_config.get("preset_filename", "Інна Гелевера.pt")
        preset_path = os.path.join(self.preset_dir, preset_file)
        if os.path.exists(preset_path):
            self.style = torch.load(preset_path, map_location=self.device)
            log.info(f"👤 [TTS] Активовано пресет: {preset_file}")

    def _text_processing_worker(self):
        while True:
            text = self.text_queue.get()
            if not text: continue
            try:
                start_tts = time.time()
                t_norm = normalize('NFKC', text.replace('+', StressSymbol.CombiningAcuteAccent))
                ps = ipa(self.stressify(t_norm))

                tokens = self.tokenizer.encode(ps)
                wav = self.engine.generate(tokens, self.speed, self.style)

                audio_chunk = wav.astype(np.float32)
                log.info(f"    🔊 [TTS згенеровано за: {time.time() - start_tts:.2f}s]")
                if not self.interrupted: self.audio_queue.put(audio_chunk)
            except Exception as e:
                log.error(f"❌ Помилка TTS: {e}")
            self.text_queue.task_done()

    def _audio_playback_worker(self):
        while True:
            audio_data = self.audio_queue.get()
            self._is_playing_now = True
            sd.play(audio_data, 24000)
            sd.wait()
            self._is_playing_now = False
            self.audio_queue.task_done()

    def play_text_async(self, text: str):
        self.text_queue.put(text)

    def is_playing(self) -> bool:
        return self._is_playing_now or not self.audio_queue.empty() or not self.text_queue.empty()

    def stop(self):
        self.interrupted = True
        sd.stop()
        while not self.text_queue.empty(): self.text_queue.get_nowait()
        while not self.audio_queue.empty(): self.audio_queue.get_nowait()
        self.interrupted = False

    def reset_session(self):
        self.interrupted = False