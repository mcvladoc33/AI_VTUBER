import os
import time
import json
from collections import deque
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


def record_microphone_clean(filename, sample_rate=16000, threshold=0.05,
                             silence_duration=0.9, pre_roll_ms=350, chunk_size=1024):
    """Послідовний запис мікрофона без фонових потоків та конфліктів заліза.

    pre_roll_ms — скільки мс "передісторії" тримати в кільцевому буфері
    ДО того, як гучність перевищить threshold. Без цього перший склад
    (м'який приголосний, видих) регулярно ковтається, бо на момент
    спрацювання порогу він уже пролетів повз raw_buffer.pop(0) і був
    відкинутий назавжди.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    audio_buffer = []
    is_speaking = False
    silence_samples = 0
    max_silence_samples = int((silence_duration * sample_rate) / chunk_size)
    raw_buffer = []

    pre_roll_chunks = max(1, int((pre_roll_ms / 1000) / (chunk_size / sample_rate)))
    pre_roll_buffer = deque(maxlen=pre_roll_chunks)

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
                        # Підхоплюємо кілька чанків "з минулого" — саме тут
                        # ховався ковтнутий початок слова
                        audio_buffer.extend(pre_roll_buffer)
                        audio_buffer.append(current_chunk)
                    else:
                        pre_roll_buffer.append(current_chunk)
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

    text_mode = config.get('text_mode', False)

    # Один зведений банер замість розкиданих по різних модулях рядків —
    # раніше LLM-потоки не показувались узагалі, а TTS-потоки друкувались
    # двічі (в TTSHandler і ще раз в ONNX-гілці)
    llm_cfg = config.get('llm', {})
    tts_cfg = config.get('tts', {})
    log.info("🖥️  [SYSTEM] Розподіл потоків CPU:")
    log.info(f"     LLM → n_threads={llm_cfg.get('n_threads', 2)} (генерація), "
             f"n_threads_batch={llm_cfg.get('n_threads_batch', 4)} (обробка запиту)")
    log.info(f"     TTS → n_threads={tts_cfg.get('n_threads', 2)} "
             f"(рушій: {str(tts_cfg.get('engine', 'pytorch')).upper()}), "
             f"parallel_chunks={tts_cfg.get('parallel_chunks', 1)}")
    log.info("-" * 60)

    # У текстовому режимі STT (Whisper) взагалі не використовується —
    # немає сенсу вантажити модель у пам'ять і платити за її ініціалізацію,
    # якщо мікрофон цього разу не потрібен
    stt_module = None if text_mode else AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)

    mic_config = config.get('mic', {})
    mic_threshold = mic_config.get('threshold', 0.05)
    mic_silence_duration = mic_config.get('silence_duration', 0.9)
    mic_pre_roll_ms = mic_config.get('pre_roll_ms', 350)

    char_name = config.get('character', {}).get('name', 'Помічниця')

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
                status = record_microphone_clean(
                    TEMP_AUDIO_PATH,
                    threshold=mic_threshold,
                    silence_duration=mic_silence_duration,
                    pre_roll_ms=mic_pre_roll_ms
                )

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

            start_llm = time.time()
            tts_module.start_turn_timer(start_llm)

            is_first_sentence = True
            last_yield_time = start_llm
            llm_total = 0.0

            for sentence in llm_module.generate_response(user_input):
                now = time.time()

                if is_first_sentence:
                    seg_time = now - start_llm
                    is_first_sentence = False
                else:
                    # Скільки LLM генерував саме ЦЕЙ відрізок (не з нуля, а
                    # від моменту, коли попередній уже пішов у TTS) — без
                    # цього не видно, чи саме LLM гальмує пізніші речення
                    seg_time = now - last_yield_time
                last_yield_time = now
                llm_total += seg_time

                # llm_time іде разом із текстом — TTSHandler сам надрукує
                # звіт по цій фразі ЖИВЦЕМ, у момент реального відтворення
                # (у _audio_playback_worker), а не постфактум одним махом
                try:
                    tts_module.play_text_async(sentence, llm_time=seg_time)
                except Exception as tts_err:
                    log.error(f" └─ ❌ [Помилка TTS]: {tts_err}")

            tts_module.wait_until_done()

            tts_total = sum(c["synth_time"] for c in tts_module.chunk_metrics)
            silence_total = sum(tts_module.silence_log)
            wall_total = time.time() - start_llm
            log.info(
                f"  📊 LLM {llm_total:.2f}s | TTS {tts_total:.2f}s | "
                f"тиша {silence_total:.2f}s | всього {wall_total:.2f}s"
            )

            tts_module.flush_session_audio()
            log.info("-" * 60)

        except KeyboardInterrupt:
            log.warning("👋 Роботу завершено. Бувай!")
            break
        except Exception as e:
            log.error(f"❌ Помилка в головному циклі: {e}")


if __name__ == "__main__":
    main()