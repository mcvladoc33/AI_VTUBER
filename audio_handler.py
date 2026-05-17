import os
from faster_whisper import WhisperModel


class AudioHandler:
    def __init__(self, config):
        stt_config = config.get('stt', {})
        print("🎙️ [STT] Завантаження Whisper на CPU...")

        self.model = WhisperModel(
            stt_config.get('model_size', 'small'),
            device=stt_config.get('device', 'cpu'),
            compute_type=stt_config.get('compute_type', 'int8')
        )
        print("✅ [STT] Whisper готовий.")

    def transcribe_audio(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return ""

        # condition_on_previous_text=False блокує повторення та "зациклення" фраз на тиші
        segments, info = self.model.transcribe(
            audio_path,
            language="uk",
            condition_on_previous_text=False,
            beam_size=5,
            no_speech_threshold=0.6
        )

        text = "".join([segment.text for segment in segments]).strip()
        return text