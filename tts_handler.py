import os, torch, numpy as np, sounddevice as sd, time as t
from logger_config import log
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier
from tts_engines import ONNXEngine
from styletts2_inference.models import StyleTTS2Tokenizer

class TTSHandler:
    def __init__(self, config):
        self.tts_config = config.get('tts', {})
        self.engine = ONNXEngine(self.tts_config.get("onnx_path", "models/styletts2_sim.onnx"), n_threads=4)
        self.tokenizer = StyleTTS2Tokenizer(hf_path="models/styletts2_ukrainian_multispeaker")
        self.stressify = Stressifier()
        self._load_preset()

    def _load_preset(self):
        path = os.path.join("voices", self.tts_config.get("preset_filename", "Інна Гелевера.pt"))
        if os.path.exists(path):
            self.style = torch.load(path, map_location='cpu')
            if hasattr(self.style, 'detach'): self.style = self.style.detach().cpu().numpy()

    def say(self, text):
        try:
            tokens = self.tokenizer.encode(ipa(self.stressify(text)))
            start_t = t.time()
            wav = self.engine.generate(tokens, self.tts_config.get("speed", 1.15), self.style)
            if wav is not None:
                sd.play(wav.astype(np.float32), samplerate=24000)
                sd.wait()
                log.info(f"    🔊 [TTS] Синтез завершено за: {t.time() - start_t:.2f}с")
        except Exception as e: log.error(f"❌ TTS Error: {e}")