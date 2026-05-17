import os
import warnings
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Приховуємо попередження від huggingface_hub
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

from faster_whisper import WhisperModel


class AudioHandler:
    def __init__(self, config):
        stt_config = config['stt']

        # Перевіряємо, чи це не велика модель, яка завалить CPU
        if stt_config['model_size'] in ['medium', 'large', 'large-v2', 'large-v3']:
            print(f"⚠️ УВАГА: Модель {stt_config['model_size']} занадто важка для твого CPU! Можливі шалені затримки.")

        print(f"📥 [STT] Перевірка та завантаження Whisper ({stt_config['model_size']}) на CPU...")

        # Завантажуємо модель (якщо її немає, вона автоматично скачається один раз)
        self.model = WhisperModel(
            stt_config['model_size'],
            device=stt_config['device'],
            compute_type=stt_config['compute_type'],
            download_root=None  # Використовує стандартний кеш системи
        )
        print("✅ [STT] Модель успішно завантажена та готова до роботи.")

        self.audio_file = "temp_voice.wav"
        self.sample_rate = 16000
        self.chunk_size = 1024

        # Налаштування чутливості
        self.threshold = 450
        self.silence_duration = 1.2
        self.max_silence_chunks = int((self.silence_duration * self.sample_rate) / self.chunk_size)

    def listen_and_record(self):
        """Постійно слухає фон і записує звук, коли користувач говорить"""
        print("\n🎤 Я тебе слухаю... (просто почни говорити)")

        audio_frames = []
        is_speaking = False
        silence_counter = 0

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16') as stream:
            while True:
                data, _ = stream.read(self.chunk_size)
                audio_data = np.frombuffer(data, dtype=np.int16)

                audio_float = audio_data.astype(np.float64)
                volume = np.sqrt(np.mean(audio_float ** 2))

                if volume > self.threshold:
                    if not is_speaking:
                        print("🎙️ Голос виявлено! Записую...")
                        is_speaking = True
                    silence_counter = 0
                    audio_frames.append(audio_data)
                else:
                    if is_speaking:
                        audio_frames.append(audio_data)
                        silence_counter += 1

                        if silence_counter > self.max_silence_chunks:
                            print("🤫 Пауза зафіксована. Обробка мовлення...")
                            break
                    else:
                        audio_frames.append(audio_data)
                        if len(audio_frames) > 25:
                            audio_frames.pop(0)

        flat_audio = np.concatenate(audio_frames)
        wavfile.write(self.audio_file, self.sample_rate, flat_audio)

    def transcribe(self) -> str:
        """Перетворює аудіофайл у текст з максимальною точністю"""
        if not os.path.exists(self.audio_file):
            return ""

        segments, _ = self.model.transcribe(
            self.audio_file,
            language="uk",
            beam_size=5,
            initial_prompt="Привіт. Це правильна, чітка українська мова без русизмів, суржику та помилок. Розпізнавай текст розмовно.",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400)
        )

        text = "".join([s.text for s in segments]).strip()

        if os.path.exists(self.audio_file):
            os.remove(self.audio_file)

        return text

    def listen_to_text(self) -> str:
        self.listen_and_record()
        return self.transcribe()