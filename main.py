import os
import sys

# Патчі пам'яті
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("input", "temp_voice.wav")


def record_microphone(filename, sample_rate=16000, threshold=0.02, silence_duration=1.2):
    # Створюємо папку (наприклад, input/), якщо її ще немає на диску
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    chunk_size = 1024
    audio_buffer = []
    is_speaking = False
    silence_samples = 0
    max_silence_samples = int((silence_duration * sample_rate) / chunk_size)
    raw_buffer = []

    def callback(indata, frames, time, status):
        raw_buffer.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=chunk_size):
        while True:
            if len(raw_buffer) > 0:
                current_chunk = raw_buffer.pop(0)
                volume_norm = np.linalg.norm(current_chunk) / np.sqrt(len(current_chunk))

                if not is_speaking:
                    if volume_norm > threshold:
                        print("\n🎙️ [Мікрофон] Запис пішов...")
                        is_speaking = True
                        audio_buffer.append(current_chunk)
                else:
                    audio_buffer.append(current_chunk)
                    if volume_norm < threshold:
                        silence_samples += 1
                    else:
                        silence_samples = 0

                    if silence_samples > max_silence_samples:
                        break
            else:
                sd.sleep(10)

    if audio_buffer:
        recording = np.concatenate(audio_buffer, axis=0)
        sf.write(filename, recording, sample_rate)
    else:
        sf.write(filename, np.zeros((sample_rate, 1)), sample_rate)


def main():
    if not os.path.exists(CONFIG_PATH):
        print("❌ Помилка: config.json не знайдено!")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Завантаження систем
    stt_module = AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)

    print("\n🚀 [SYSTEM] Помічниця повністю готова до роботи!")
    print("--------------------------------------------------")

    while True:
        try:
            print("🟢 Очікую ваш голос...")
            record_microphone(TEMP_AUDIO_PATH, threshold=0.02, silence_duration=1.2)

            user_input = stt_module.transcribe_audio(TEMP_AUDIO_PATH)

            if not user_input or len(user_input.strip()) < 2:
                continue

            print(f"👤 Ви: {user_input}")

            # Генерація думки
            response_text = llm_module.generate_response(user_input)
            print(f"🤖 {config['character']['name']}: {response_text}")

            # Озвучування
            tts_module.generate_speech(response_text)
            print("--------------------------------------------------")

        except KeyboardInterrupt:
            print("\n👋 Роботу завершено. Бувай!")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    main()