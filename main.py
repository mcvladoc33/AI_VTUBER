import os
import sys
import asyncio
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import warnings
import logging

# --- ПРИХОВУЄМО ТЕХНІЧНИЙ ШУМ ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("torch").setLevel(logging.ERROR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from logger_config import log
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("inputs", "temp_voice.wav")

stt_to_llm_queue = asyncio.Queue()
llm_to_tts_queue = asyncio.Queue()

stop_event = asyncio.Event()
interrupt_event = asyncio.Event()

STOP_WORDS = ["стоп", "stop", "досить", "зупинись", "замовкни", "харе", "поп", "порохуй"]


def record_microphone_core(filename, sample_rate=16000, threshold=0.033, silence_duration=1.0):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    chunk_size = 1024
    audio_buffer = []
    is_speaking = False
    silence_samples = 0
    max_silence_samples = int((silence_duration * sample_rate) / chunk_size)

    pre_record_len = int((0.4 * sample_rate) / chunk_size)
    pre_record_buffer = []
    raw_buffer = []

    def callback(indata, frames, time_info, status):
        raw_buffer.append(indata.copy())

    import time as t
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
                t.sleep(0.01)

    if audio_buffer and not stop_event.is_set():
        recording = np.concatenate(audio_buffer, axis=0)
        sf.write(filename, recording, sample_rate)
        return True
    return False


async def input_stt_worker(text_mode, stt_module, tts_module):
    log.info("🟢 [Задача 1: Введення/STT] Успішно запущено.")
    while not stop_event.is_set():
        try:
            if text_mode:
                user_input = await asyncio.to_thread(lambda: input("👤 Ви: ").strip())
                if not user_input: continue
                if user_input.lower() in STOP_WORDS:
                    interrupt_event.set()
                    await asyncio.to_thread(tts_module.stop)
                    continue
                await stt_to_llm_queue.put(user_input)
            else:
                recorded = await asyncio.to_thread(record_microphone_core, TEMP_AUDIO_PATH, 16000, 0.033, 1.0)
                if recorded:
                    if stop_event.is_set(): break
                    user_input = await asyncio.to_thread(stt_module.transcribe_audio, TEMP_AUDIO_PATH)
                    if not user_input or len(user_input.strip()) < 2: continue

                    clean_text = user_input.lower().strip().replace(".", "").replace("!", "").replace(",", "")
                    if any(word in clean_text for word in STOP_WORDS):
                        interrupt_event.set()
                        await asyncio.to_thread(tts_module.stop)
                        while not stt_to_llm_queue.empty(): stt_to_llm_queue.get_nowait()
                        while not llm_to_tts_queue.empty(): llm_to_tts_queue.get_nowait()
                        continue

                    if await asyncio.to_thread(tts_module.is_playing): continue
                    log.info(f"\n👤 Ви: {user_input}")
                    await stt_to_llm_queue.put(user_input)
                else:
                    await asyncio.sleep(0.1)
        except Exception as e:
            log.error(f"❌ Помилка в Задачі STT: {e}")
            await asyncio.sleep(1)


def get_next_sentence(generator):
    try:
        return next(generator)
    except StopIteration:
        return None


async def llm_processing_worker(llm_module, char_name):
    log.info("🟢 [Задача 2: Обробка/LLM] Успішно запущено.")
    while not stop_event.is_set():
        try:
            user_input = await stt_to_llm_queue.get()
            interrupt_event.clear()
            log.info(f"🧠 [LLM] {char_name} формує відповідь...")
            import time as t
            start_llm = t.time()
            is_first = True
            gen = llm_module.generate_response(user_input, interrupt_event=interrupt_event)

            while True:
                if interrupt_event.is_set() or stop_event.is_set(): break
                result = await asyncio.to_thread(get_next_sentence, gen)
                if result is None: break
                sentence, gen_time = result
                if is_first:
                    log.info(f" ⏱️ [Перший токен через: {t.time() - start_llm:.2f}s]")
                    is_first = False
                log.info(f"  ➔ {sentence} [Генерація: {gen_time:.2f}s]")
                await llm_to_tts_queue.put(sentence)
            stt_to_llm_queue.task_done()
        except Exception as e:
            log.error(f"❌ Помилка в Задачі LLM: {e}")
            await asyncio.sleep(1)


async def tts_pipeline_worker(tts_module):
    log.info("🟢 [Задача 3: Конвеєр Синтезу] Успішно запущено.")
    while not stop_event.is_set():
        try:
            sentence = await llm_to_tts_queue.get()
            if interrupt_event.is_set():
                llm_to_tts_queue.task_done()
                continue
            if len(sentence.strip()) < 6:
                llm_to_tts_queue.task_done()
                continue
            await asyncio.to_thread(tts_module.reset_session)
            await asyncio.to_thread(tts_module.play_text_async, sentence)
            llm_to_tts_queue.task_done()
        except Exception as e:
            log.error(f"❌ Помилка в Задачах конвеєра TTS: {e}")
            await asyncio.sleep(1)


async def main_async():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        log.info("🚀 [SYSTEM] Процесу AI_VTUBER присвоєно ВИСОКИЙ пріоритет.")
    except:
        pass

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    stt_module = AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)

    tasks = [
        asyncio.create_task(input_stt_worker(config.get('text_mode', False), stt_module, tts_module)),
        asyncio.create_task(llm_processing_worker(llm_module, config.get('character', {}).get('name', 'Селті'))),
        asyncio.create_task(tts_pipeline_worker(tts_module))
    ]

    log.info("🚀 [SYSTEM] Роботу конвеєра стабілізовано. Можна починати діалог!")
    try:
        await stop_event.wait()
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        stop_event.set()
        await asyncio.to_thread(tts_module.stop)
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        log.info("👋 Усі асинхронні задачі успішно закриті. Бувай!")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass