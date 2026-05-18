import os
from faster_whisper import WhisperModel


class AudioHandler:
    def __init__(self, config):
        stt_config = config.get('stt', {})
        print(f"🎙️ [STT] Завантаження Whisper ({stt_config.get('model_size', 'small')}) на CPU...")

        self.model = WhisperModel(
            stt_config.get('model_size', 'small'),
            device=stt_config.get('device', 'cpu'),
            compute_type=stt_config.get('compute_type', 'int8')
        )
        print("✅ [STT] Whisper готовий.")

    def transcribe_audio(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return ""

        # Короткий нейтральний промпт, щоб модель знала ім'я, але не вигадувала казки на стукоті
        streamer_prompt = "Селті, привіт! Що там, як справи, розкажи."

        # condition_on_previous_text=False блокує повторення та "зациклення" фраз на тиші
        segments, info = self.model.transcribe(
            audio_path,
            language="uk",
            condition_on_previous_text=False,
            beam_size=5,
            best_of=5,  # Шукає найкращий варіант з 5 спроб
            temperature=0.0,  # Модель максимально "сувора", щоб не вигадувати слова з шуму
            initial_prompt=streamer_prompt,  # Задаємо словниковий орієнтир
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4  # Жорсткий захист від спаму однаковими словами
        )

        text = "".join([segment.text for segment in segments]).strip()

        # Додаткова страховка: фікс зайвих пробілів перед знаками запитання чи крапками
        text = text.replace(" ?", "?").replace(" !", "!").replace(" .", ".")

        return text