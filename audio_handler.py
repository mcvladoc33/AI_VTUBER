import os
import re
from faster_whisper import WhisperModel
from logger_config import log

class AudioHandler:
    def __init__(self, config):
        stt_config = config.get('stt', {})
        model_size = stt_config.get('model_size', 'small')
        device = stt_config.get('device', 'cpu')
        compute_type = stt_config.get('compute_type', 'int8')

        log.info(f"🎙️ [STT] Завантаження Whisper ({model_size}) на {device.upper()} (Квантування: {compute_type})...")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        log.info("✅ [STT] Whisper успішно ініціалізовано.")

    def transcribe_audio(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return ""

        try:
            # Повертаємо перевірені параметри, які працювали стабільно
            segments, info = self.model.transcribe(
                audio_path,
                language="uk",
                beam_size=1,
                vad_filter=True,
                # Стандартні, надійні параметри VAD для Whisper
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=500
                ),
                temperature=0.0
            )

            text = " ".join([segment.text for segment in segments]).strip()
            text = re.sub(r'\s+', ' ', text)

            # Базова фільтрація сміття
            ignore_list = ["розмовна українська мова", "чіткі репліки", "без галюцинацій"]
            if len(text) < 3 or any(phrase in text.lower() for phrase in ignore_list):
                return ""

            return text

        except Exception as e:
            log.error(f"❌ Помилка розпізнавання: {e}")
            return ""