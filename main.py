import os
import sys
import warnings
import builtins

# --- 1. ТОТАЛЬНЕ ПРИДУШЕННЯ ВАРНІНГІВ ТА СИСТЕМНИХ ЛОГІВ ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="torch")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 2. БРОНЕБІЙНИЙ ПАТЧ КОДУВАННЯ (Рятує від UnicodeDecodeError) ---
original_open = builtins.open


def robust_utf8_open(file, mode='r', *args, **kwargs):
    if 'b' not in mode:
        kwargs.setdefault('encoding', 'utf-8')
    return original_open(file, mode, *args, **kwargs)


builtins.open = robust_utf8_open

# --- 3. ПАТЧ ДЛЯ HUGGINGFACE ТА МОДЕЛЕЙ STYLETTS2 ---
import styletts2_inference.models

styletts2_inference.models.open = robust_utf8_open


def fake_hf_hub_download(repo_id, filename, **kwargs):
    if os.path.exists(repo_id):
        return os.path.join(repo_id, filename)
    from huggingface_hub import hf_hub_download as original_hf_hub_download
    return original_hf_hub_download(repo_id, filename, **kwargs)


styletts2_inference.models.hf_hub_download = fake_hf_hub_download

# --- 4. ТЕПЕР ІМПОРТУЄМО АСИНХРОННІСТЬ ТА ІНШІ БІБЛІОТЕКИ ---
import asyncio
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import psutil
import time as t

# --- 5. ІМПОРТ МОДУЛІВ ПРОЄКТУ ---
from logger_config import log
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("inputs", "temp_voice.wav")
# Залишено лише надійні стоп-слова без хибних спрацьовувань
STOP_WORDS = ["стоп", "stop", "замовкни"]

stt_to_llm = asyncio.Queue()
llm_to_tts = asyncio.Queue()
stop_ev = asyncio.Event()
inter_ev = asyncio.Event()


# --- 6. ФУНКЦІЯ ЗАПИСУ З МІКРОФОНА (VAD) ---
def record_microphone_core(filename, sample_rate=16000, threshold=0.035, silence_duration=1.2):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    chunk_size = 1024
    audio_buffer, is_speaking = [], False
    silence_samples, max_silence_samples = 0, int((silence_duration * sample_rate) / chunk_size)
    pre_record_len, pre_record_buffer, raw_buffer = int((0.4 * sample_rate) / chunk_size), [], []

    def callback(indata, frames, time_info, status):
        raw_buffer.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, blocksize=chunk_size):
        while not stop_ev.is_set():
            if raw_buffer:
                current_chunk = raw_buffer.pop(0)
                volume_norm = np.linalg.norm(current_chunk) / np.sqrt(len(current_chunk))
                if not is_speaking:
                    pre_record_buffer.append(current_chunk)
                    if len(pre_record_buffer) > pre_record_len: pre_record_buffer.pop(0)
                    if volume_norm > threshold:
                        is_speaking = True
                        audio_buffer.extend(pre_record_buffer)
                        audio_buffer.append(current_chunk)
                else:
                    audio_buffer.append(current_chunk)
                    silence_samples = silence_samples + 1 if volume_norm < threshold else 0
                    if silence_samples > max_silence_samples: break
            else:
                t.sleep(0.01)

    if audio_buffer and not stop_ev.is_set():
        sf.write(filename, np.concatenate(audio_buffer, axis=0), sample_rate)
        return True
    return False


# --- 7. АСИНХРОННІ ВОРКЕРИ ---
async def input_stt_worker(text_mode, stt_module, tts_module):
    log.info("🟢 [Задача 1: Введення/STT] Успішно запущено.")
    while not stop_ev.is_set():
        try:
            if text_mode:
                user_input = await asyncio.to_thread(lambda: input("👤 Ви: ").strip())
                if not user_input: continue
                if user_input.lower() in STOP_WORDS:
                    inter_ev.set()
                    tts_module.stop()
                    continue
                await stt_to_llm.put(user_input)
            else:
                recorded = await asyncio.to_thread(record_microphone_core, TEMP_AUDIO_PATH, 16000, 0.035, 2.2)
                if recorded and not stop_ev.is_set():
                    user_input = await asyncio.to_thread(stt_module.transcribe_audio, TEMP_AUDIO_PATH)
                    if not user_input or len(user_input.strip()) < 2: continue

                    if any(word in user_input.lower() for word in STOP_WORDS):
                        inter_ev.set()
                        tts_module.stop()
                        while not stt_to_llm.empty(): stt_to_llm.get_nowait()
                        while not llm_to_tts.empty(): llm_to_tts.get_nowait()
                        log.info("🛑 [SYSTEM] Діалог перервано користувачем. Черги очищено.")
                        continue

                    log.info(f"\n👤 Ви: {user_input}")
                    await stt_to_llm.put(user_input)
        except Exception as e:
            log.error(f"❌ Помилка в Задачі STT: {e}")
            await asyncio.sleep(1)


async def llm_worker(llm):
    log.info("🟢 [Задача 2: Обробка/LLM] Успішно запущено.")
    while not stop_ev.is_set():
        try:
            user_input = await stt_to_llm.get()
            inter_ev.clear()
            gen = llm.generate_response(user_input, interrupt_event=inter_ev)

            while True:
                if inter_ev.is_set() or stop_ev.is_set(): break

                if llm_to_tts.qsize() >= 1:
                    await asyncio.sleep(0.05)
                    continue

                res = await asyncio.to_thread(lambda: next(gen, None))
                if res is None: break

                text_chunk, gen_time = res
                log.info(f"  ➔ {text_chunk} [LLM Gen: {gen_time:.2f}s]")
                await llm_to_tts.put(text_chunk)

            stt_to_llm.task_done()
        except Exception as e:
            log.error(f"❌ Помилка в Задачі LLM: {e}")
            await asyncio.sleep(1)


async def tts_worker(tts):
    log.info("🟢 [Задача 3: Конвеєр Синтезу] Успішно запущено.")
    while not stop_ev.is_set():
        try:
            sentence = await llm_to_tts.get()
            if not inter_ev.is_set():
                start = t.time()
                await asyncio.to_thread(tts.say, sentence)
                log.info(f"  🔊 [PIPELINE] Фраза повністю відпрацьована за: {t.time() - start:.4f}s")
            llm_to_tts.task_done()
        except Exception as e:
            log.error(f"❌ Помилка в Задачі TTS: {e}")
            await asyncio.sleep(1)


async def main_async():
    try:
        psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS)
    except:
        pass

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    stt_module = AudioHandler(cfg)
    llm = LLMHandler(cfg)
    tts_module = TTSHandler(cfg)

    tasks = [
        asyncio.create_task(input_stt_worker(cfg.get('text_mode', False), stt_module, tts_module)),
        asyncio.create_task(llm_worker(llm)),
        asyncio.create_task(tts_worker(tts_module))
    ]

    log.info("🚀 [SYSTEM] Роботу конвеєра стабілізовано!")
    await stop_ev.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        stop_ev.set()
        log.info("Зупинка системи...")