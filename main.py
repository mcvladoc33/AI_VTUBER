import os
import time
import json
import numpy as np
import sounddevice as sd
import soundfile as sf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from logger_config import log
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("input", "temp_voice.wav")


def record_microphone_clean(filename, sample_rate=16000, threshold=0.05, silence_duration=1.6):
    """Послідовний запис мікрофона без фонових потоків та конфліктів заліза"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    chunk_size = 1024
    audio_buffer = []
    is_speaking = False
    silence_samples = 0
    max_silence_samples = int((silence_duration * sample_rate) / chunk_size)
    raw_buffer = []

    def callback(indata, frames, time_info, status):
        raw_buffer.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=chunk_size):
        while True:
            if len(raw_buffer) > 0:
                current_chunk = raw_buffer.pop(0)
                volume_norm = np.linalg.norm(current_chunk) / np.sqrt(len(current_chunk))

                if not is_speaking:
                    if volume_norm > threshold:
                        log.info("🎙️ [Мікрофон] Запис пішов...")
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
        log.critical(f"❌ Конфігураційний файл {CONFIG_PATH} відсутній!")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    stt_module = AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)

    char_name = config.get('character', {}).get('name', 'Помічниця')
    text_mode = config.get('text_mode', False)

    log.info("🚀 [SYSTEM] Помічниця повністю готова до роботи!")
    if text_mode:
        log.info("👉 Режим: ТЕКСТОВИЙ (вводь текст у консоль та тисни Enter).")
    else:
        log.info("👉 Режим: МІКРОФОН (просто починай говорити, коли з'явиться індикатор).")
    log.info("-" * 60)

    while True:
        try:
            tts_module.reset_session()

            if text_mode:
                log.info("🟢 Очікую ваш текст...")
                user_input = input("👤 Ви: ").strip()
                if not user_input:
                    continue
            else:
                log.info("🟢 Очікую ваш голос...")
                status = record_microphone_clean(TEMP_AUDIO_PATH)

                if status == "AUDIO_RECORDED":
                    start_stt = time.time()
                    user_input = stt_module.transcribe_audio(TEMP_AUDIO_PATH)
                    stt_time = time.time() - start_stt

                    if not user_input or len(user_input.strip()) < 2:
                        continue
                    log.info(f"👤 Ви: {user_input} [STT: {stt_time:.2f}s]")
                else:
                    continue

            log.info(f"🤖 {char_name}:")

            is_first_sentence = True
            start_llm = time.time()

            for sentence in llm_module.generate_response(user_input):
                if is_first_sentence:
                    llm_first_token_time = time.time() - start_llm
                    log.info(f" ⏱️ [Пошук думки: {llm_first_token_time:.2f}s]")
                    is_first_sentence = False

                log.info(f" ➔ {sentence}")

                try:
                    tts_module.play_text_async(sentence)
                except Exception as tts_err:
                    log.error(f" └─ ❌ [Помилка TTS]: {tts_err}")

            tts_module.wait_until_done()
            tts_module.flush_session_audio()
            log.info("-" * 60)

        except KeyboardInterrupt:
            log.warning("👋 Роботу завершено. Бувай!")
            break
        except Exception as e:
            log.error(f"❌ Помилка в головному циклі: {e}")


if __name__ == "__main__":
    main()