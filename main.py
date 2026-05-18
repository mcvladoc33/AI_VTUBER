import os
import sys
import time
import json
import numpy as np
import sounddevice as sd
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("input", "temp_voice.wav")


def record_microphone_clean(filename, sample_rate=16000, threshold=0.05, silence_duration=1.2):
    """Класичний послідовний запис мікрофона без фонових потоків та конфліктів заліза"""
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
        return "AUDIO_RECORDED"
    return "EMPTY"


def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Помилка: {CONFIG_PATH} не знайдено!")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    stt_module = AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)

    char_name = config.get('character', {}).get('name', 'Помічниця')

    # Зчитуємо режим роботи прямо з конфігу
    text_mode = config.get('text_mode', False)

    print("\n🚀 [SYSTEM] Помічниця повністю готова до роботи!")
    if text_mode:
        print("👉 Режим: ТЕКСТОВИЙ (вводь текст у консоль та тисни Enter).")
    else:
        print("👉 Режим: МІКРОФОН (просто починай говорити, коли з'явиться індикатор).")
    print("--------------------------------------------------")

    while True:
        try:
            tts_module.reset_session()

            if text_mode:
                print("🟢 Очікую ваш текст...")
                user_input = input("👤 Ви: ").strip()
                if not user_input:
                    continue
            else:
                print("🟢 Очікую ваш голос...")
                status = record_microphone_clean(TEMP_AUDIO_PATH, threshold=0.05, silence_duration=1.2)

                if status == "AUDIO_RECORDED":
                    start_stt = time.time()
                    user_input = stt_module.transcribe_audio(TEMP_AUDIO_PATH)
                    stt_time = time.time() - start_stt

                    if not user_input or len(user_input.strip()) < 2:
                        continue
                    print(f"👤 Ви: {user_input} [STT: {stt_time:.2f}s]")
                else:
                    continue

            print(f"🤖 {char_name}:")

            is_first_sentence = True
            start_llm = time.time()

            for sentence in llm_module.generate_response(user_input):
                if is_first_sentence:
                    llm_first_token_time = time.time() - start_llm
                    print(f" ⏱️ [Пошук думки: {llm_first_token_time:.2f}s]")
                    is_first_sentence = False

                print(f" ➔ {sentence}")

                start_tts = time.time()
                try:
                    tts_module.play_text_async(sentence)
                    tts_time = time.time() - start_tts
                    print(f"   └─ 🔊 [Речення передано в TTS за: {tts_time:.4f}s]")
                except Exception as tts_err:
                    print(f"   └─ ❌ [Помилка TTS]: {tts_err}")

            # Чекаємо, поки Селті повністю договорить речення в плеєрі
            tts_module.wait_until_done()
            print("\n--------------------------------------------------")

        except KeyboardInterrupt:
            print("\n👋 Роботу завершено. Бувай!")
            break
        except Exception as e:
            print(f"❌ Помилка в головному циклі: {e}")


if __name__ == "__main__":
    main()