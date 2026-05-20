import os
import sys
import time
import json
import queue
import threading
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

stt_to_llm_queue = queue.Queue()
llm_to_tts_queue = queue.Queue()

stop_event = threading.Event()
interrupt_event = threading.Event()

# Список слів, які мають право примусово перебити або зупинити Селті
STOP_WORDS = ["стоп", "stop", "досить", "зупинись", "замовкни", "харе", "поп", "порохуй"]


def record_microphone_core(filename, sample_rate=16000, threshold=0.033, silence_duration=1.5):
    """
    Запис мікрофону. Більше НЕ перериває TTS автоматично при будь-якому звуці.
    Селті продовжує говорити, поки мікрофон фоном слухає команду СТОП.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    chunk_size = 1024
    audio_buffer = []
    is_speaking = False
    silence_samples = 0
    max_silence_samples = int((silence_duration * sample_rate) / chunk_size)

    pre_record_len = int((0.4 * sample_rate) / chunk_size)
    pre_record_buffer = []
    raw_buffer = []

    def callback(indata, frames, time, status):
        raw_buffer.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=chunk_size):
        while not stop_event.is_set():
            if len(raw_buffer) > 0:
                current_chunk = raw_buffer.pop(0)
                volume_norm = np.linalg.norm(current_chunk) / np.sqrt(len(current_chunk))

                if not is_speaking:
                    pre_record_buffer.append(current_chunk)
                    if len(pre_record_buffer) > pre_record_len:
                        pre_record_buffer.pop(0)

                    if volume_norm > threshold:
                        is_speaking = True
                        if pre_record_buffer:
                            audio_buffer.extend(pre_record_buffer)
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
                time.sleep(0.01)

    if audio_buffer and not stop_event.is_set():
        recording = np.concatenate(audio_buffer, axis=0)
        sf.write(filename, recording, sample_rate)
        return True
    return False


def input_stt_worker(text_mode, stt_module, tts_module):
    log.info("🟢 [Потік 1: Введення/STT] Успішно запущено.")
    while not stop_event.is_set():
        try:
            if text_mode:
                user_input = input("👤 Ви: ").strip()
                if not user_input: continue

                if user_input.lower() in STOP_WORDS:
                    log.warning("🛑 [SYSTEM] Отримано текстову команду СТОП.")
                    interrupt_event.set()
                    tts_module.stop()
                    continue

                stt_to_llm_queue.put(user_input)
            else:
                # Викликаємо ядро запису (воно тепер пасивне, не гасить звук само по собі)
                if record_microphone_core(TEMP_AUDIO_PATH, threshold=0.033, silence_duration=1.5):
                    if stop_event.is_set(): break

                    start_stt = time.time()
                    user_input = stt_module.transcribe_audio(TEMP_AUDIO_PATH)
                    stt_time = time.time() - start_stt

                    if not user_input or len(user_input.strip()) < 2:
                        continue

                    # Перевіряємо, чи розпізнаний текст містить команду СТОП
                    clean_text = user_input.lower().strip().replace(".", "").replace("!", "").replace(",", "")
                    is_stop_command = any(word in clean_text for word in STOP_WORDS)

                    if is_stop_command:
                        log.warning(f"🛑 [SYSTEM] Перехоплено команду СТОП ('{user_input}'). Скидання конвеєра.")
                        interrupt_event.set()
                        tts_module.stop()

                        # Моментально чистимо беклог черг
                        while not stt_to_llm_queue.empty():
                            try:
                                stt_to_llm_queue.get_nowait(); stt_to_llm_queue.task_done()
                            except:
                                break
                        while not llm_to_tts_queue.empty():
                            try:
                                llm_to_tts_queue.get_nowait(); llm_to_tts_queue.task_done()
                            except:
                                break
                        continue

                    # Якщо Селті щось активно говорила в цей момент, а ми сказали НЕ команду стоп —
                    # ми просто ігноруємо цей ввід як випадковий шум/перебивання, щоб не збивати її з думки.
                    if tts_module.is_playing():
                        continue

                    log.info(f"\n👤 Ви: {user_input} [STT: {stt_time:.2f}s]")
                    stt_to_llm_queue.put(user_input)
        except Exception as e:
            log.error(f"❌ Помилка в Потоці STT: {e}")
            time.sleep(1)


def llm_processing_worker(llm_module, char_name):
    log.info("🟢 [Потік 2: Обробка/LLM] Успішно запущено.")
    while not stop_event.is_set():
        try:
            try:
                user_input = stt_to_llm_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            interrupt_event.clear()

            log.info(f"🧠 [LLM] {char_name} формує відповідь...")
            start_llm = time.time()
            is_first = True

            for sentence in llm_module.generate_response(user_input, interrupt_event=interrupt_event):
                if interrupt_event.is_set() or stop_event.is_set():
                    break

                if is_first:
                    log.info(f" ⏱️ [Перший токен через: {time.time() - start_llm:.2f}s]")
                    is_first = False

                log.info(f"  ➔ {sentence}")
                llm_to_tts_queue.put(sentence)

            stt_to_llm_queue.task_done()
        except Exception as e:
            log.error(f"❌ Помилка в Потоці LLM: {e}")
            time.sleep(1)


def tts_render_worker(tts_module):
    log.info("🟢 [Потік 3: Озвучка/TTS] Успішно запущено.")
    while not stop_event.is_set():
        try:
            try:
                sentence = llm_to_tts_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if interrupt_event.is_set():
                llm_to_tts_queue.task_done()
                continue

            try:
                tts_module.reset_session()
                tts_module.play_text_async(sentence)

                while tts_module.is_playing():
                    if interrupt_event.is_set() or stop_event.is_set():
                        tts_module.stop()
                        break
                    time.sleep(0.02)

            except Exception as tts_err:
                log.error(f"❌ Помилка синтезу мовлення: {tts_err}")

            llm_to_tts_queue.task_done()
        except Exception as e:
            log.error(f"❌ Помилка в Потоці TTS: {e}")
            time.sleep(1)


def main():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        log.info("🚀 [SYSTEM] Процесу AI_VTUBER присвоєно ВИСОКИЙ пріоритет у системі Windows.")
    except:
        pass

    if not os.path.exists(CONFIG_PATH):
        log.critical(f"❌ Конфігураційний файл {CONFIG_PATH} відсутній!")
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    stt_module = AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)

    char_name = config.get('character', {}).get('name', 'Селті')
    text_mode = config.get('text_mode', False)

    log.info("\n🚀 [SYSTEM] Запуск асинхронних конвеєрів...")
    log.info("-" * 60)

    t1 = threading.Thread(target=input_stt_worker, args=(text_mode, stt_module, tts_module), daemon=True)
    t2 = threading.Thread(target=llm_processing_worker, args=(llm_module, char_name), daemon=True)
    t3 = threading.Thread(target=tts_render_worker, args=(tts_module,), daemon=True)

    t1.start()
    t2.start()
    t3.start()

    log.info("🚀 [SYSTEM] Роботу конвеєра стабілізовано. Можна починати діалог!")
    log.info("-" * 60)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.warning("\n👋 Завершення роботи програми...")
        stop_event.set()
        tts_module.stop()
        time.sleep(0.5)
        log.info("👋 Усі потоки успішно закрито. Бувай!")


if __name__ == "__main__":
    main()