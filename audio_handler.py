import os
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
        log.info("✅ [STT] Whisper успішно ініціалізовано та готовий до роботи.")

    def transcribe_audio(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return ""

        # 🔥 Сухий технічний промпт захищає від появи лівих фраз під час зітхань або шуму
        technical_prompt = "Розмовна українська мова, чіткі репліки без галюцинацій."

        segments, info = self.model.transcribe(
            audio_path,
            language="uk",
            condition_on_previous_text=False,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            initial_prompt=technical_prompt,
            no_speech_threshold=0.6,  # Трохи підняли поріг відсікання тиші/шуму
            compression_ratio_threshold=2.4
        )

        text = "".join([segment.text for segment in segments]).strip()
        text = text.replace(" ?", "?").replace(" !", "!").replace(" .", ".")

        # Жорсткий чорний список для фраз-привидів
        hallucination_blacklist = [
            "дякую за перегляд",
            "продовження випливає",
            "субтитри",
            "редактор",
            "підписуйтесь",
            "бувай",
            "що там як справи",
            "що там як справи розкажи",
            "розмовна українська мова"
        ]

        clean_check = text.lower().strip().replace(".", "").replace("!", "").replace("?", "").replace(",", "")

        if len(clean_check) <= 2 or clean_check in hallucination_blacklist:
            return ""

        return text