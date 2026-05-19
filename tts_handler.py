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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from logger_config import log
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol
import styletts2_inference.models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import builtins
import yaml

original_open = builtins.open


def robust_utf8_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)


styletts2_inference.models.open = robust_utf8_open
yaml.safe_load_original = yaml.safe_load


def patch_yaml_load(stream, *args, **kwargs):
    if isinstance(stream, str):
        with robust_utf8_open(stream, 'r', encoding='utf-8') as f:
            return yaml.safe_load_original(f, *args, **kwargs)
    return yaml.safe_load_original(stream, *args, **kwargs)


yaml.safe_load = patch_yaml_load
builtins.open = robust_utf8_open


def fake_hf_hub_download(repo_id, filename, **kwargs):
    return os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker", filename)


styletts2_inference.models.hf_hub_download = fake_hf_hub_download


class TTSHandler:
    def __init__(self, config):
        self.config = config
        self.tts_config = config.get('tts', {})

        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self.styletts_path = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
        self.preset_dir = os.path.join(BASE_DIR, "voices")
        self.output_dir = os.path.join(BASE_DIR, "outputs")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f"📥 [TTS] Ініціалізація StyleTTS2 UA на девайсі: {self.device.upper()}")

        self.speed = self.tts_config.get("speed", 1.15)
        self.noise_scale = self.tts_config.get("noise_scale", 0.05)

        self.interrupted = False
        self._is_playing_now = False

        from styletts2_inference.models import StyleTTS2
        self.multi_model = StyleTTS2(hf_path=self.styletts_path, device=self.device)

        try:
            if hasattr(self.multi_model, 'model') and self.multi_model.model is not None:
                self.multi_model.model.diffusion_steps = 1
                if hasattr(self.multi_model.model, 'sampler'):
                    self.multi_model.model.sampler.steps = 1
        except Exception as e:
            log.warning(f"⚠️ Не вдалося примусово зрізати кроки дифузії: {e}")

        self.stressify = Stressifier()
        self.ipa_func = ipa
        self.style = None
        self._load_preset()

        threading.Thread(target=self._text_processing_worker, daemon=True).start()
        threading.Thread(target=self._audio_playback_worker, daemon=True).start()

    def _load_preset(self):
        preset_file = self.tts_config.get("preset_filename", "Інна Гелевера.pt")
        preset_path = os.path.join(self.preset_dir, preset_file)
        if os.path.exists(preset_path):
            self.style = torch.load(preset_path, map_location=self.device)
            log.info(f"👤 [TTS] Активовано пресет голосу: {preset_file}")
        else:
            log.error(f"❌ [TTS] Файл пресету не знайдено: {preset_path}")

    def _text_processing_worker(self):
        while True:
            text = self.text_queue.get()
            if text is None: continue

            if self.interrupted or not text.strip():
                self.text_queue.task_done()
                continue

            try:
                start_tts = time.time()
                t_norm = normalize('NFKC', text.replace('+', StressSymbol.CombiningAcuteAccent))
                ps = self.ipa_func(self.stressify(t_norm))

                if ps and not self.interrupted:
                    tokens = self.multi_model.tokenizer.encode(ps)
                    current_style = self.style.clone() if self.style is not None else None
                    wav = self.multi_model(tokens, speed=self.speed, s_prev=current_style)

                    if self.interrupted:
                        self.text_queue.task_done()
                        continue

                    audio_chunk = wav.cpu().numpy().flatten()
                    log.info(f"    🔊 [TTS згенеровано за: {time.time() - start_tts:.2f}s] ➔ Склади: {len(tokens)}")

                    if not self.interrupted:
                        self.audio_queue.put(audio_chunk)
            except Exception as e:
                log.error(f"❌ Помилка всередині обробника тексту TTS: {e}")

            self.text_queue.task_done()

    def _audio_playback_worker(self):
        while True:
            audio_data = self.audio_queue.get()
            if audio_data is None: continue
            if self.interrupted:
                self.audio_queue.task_done()
                continue
            try:
                self._is_playing_now = True
                sd.play(audio_data, 24000)

                # Повертаємо класичний, стабільний контроль потоку
                while not self.interrupted:
                    try:
                        if not sd.get_stream().active:
                            break
                    except:
                        break
                    time.sleep(0.005)

                if self.interrupted:
                    sd.stop()
            except:
                pass
            finally:
                self._is_playing_now = False
                self.audio_queue.task_done()

    def play_text_async(self, text: str):
        if self.interrupted: return
        self.text_queue.put(text)

    def is_audio_playing(self) -> bool:
        try:
            return sd._get_stream_status() or self._is_playing_now
        except:
            return self._is_playing_now

    def is_playing(self) -> bool:
        return self.is_audio_playing() or not self.audio_queue.empty() or not self.text_queue.empty()

    def stop(self):
        self.interrupted = True
        try:
            sd.stop()
        except:
            pass
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait(); self.text_queue.task_done()
            except queue.Empty:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait(); self.audio_queue.task_done()
            except queue.Empty:
                break
        self._is_playing_now = False
        log.warning("🛑 [TTS] Усі звукові черги повністю очищено та зупинено.")

    def reset_session(self):
        self.interrupted = False