import os
import sys
import time

# 🔥 Збільшуємо кількість потоків до 4. Оскільки mBART вимкнено в config.json,
# ядра процесора більше не конфліктують, а віддають всю потужність на 50 кроків StyleTTS2.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

# Налаштовуємо PyTorch на використання фізичних ядер вашого CPU
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("input", "temp_voice.wav")


def record_microphone(filename, sample_rate=16000, threshold=0.02, silence_duration=1.2):
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

    stt_module = AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)

    char_name = config.get('character', {}).get('name', 'Помічниця')

    print("\n🚀 [SYSTEM] Помічниця повністю готова до роботи!")
    print("--------------------------------------------------")

    while True:
        try:
            print("🟢 Очікую ваш голос...")
            record_microphone(TEMP_AUDIO_PATH, threshold=0.02, silence_duration=1.2)

            start_stt = time.time()
            user_input = stt_module.transcribe_audio(TEMP_AUDIO_PATH)
            stt_time = time.time() - start_stt

            if not user_input or len(user_input.strip()) < 2:
                continue

            print(f"👤 Ви: {user_input} [STT: {stt_time:.2f}s]")
            print(f"🤖 {char_name}:")

            is_first_sentence = True
            start_llm = time.time()

            for sentence in llm_module.generate_response(user_input):
                if is_first_sentence:
                    llm_first_token_time = time.time() - start_llm
                    print(f" ⏱️ [Пошук думки: {llm_first_token_time:.2f}s]")
                    is_first_sentence = False

                # Друкуємо речення на екран
                print(f" ➔ {sentence}")

                # Озвучуємо речення через виправлений tts_handler
                start_tts = time.time()
                try:
                    tts_module.generate_speech(sentence)
                    tts_time = time.time() - start_tts
                    print(f"   └─ 🔊 [Синтез голосу за: {tts_time:.2f}s]")
                except Exception as tts_err:
                    print(f"   └─ ❌ [Помилка TTS]: {tts_err}")

            print("\n--------------------------------------------------")

        except KeyboardInterrupt:
            print("\n👋 Роботу завершено. Бувай!")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    main()