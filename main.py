import os
import sys
import warnings

# --- ЯДЕРНЕ ПРИДУШЕННЯ (прибирає сміття від llama.cpp/torch при старті) ---
class SilenceStream:
    def write(self, *args, **kwargs): pass
    def flush(self): pass
_original_stdout, _original_stderr = sys.stdout, sys.stderr
sys.stdout = sys.stderr = SilenceStream()

import builtins
import asyncio
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import psutil
import time as t

# --- ПАТЧІ ---
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

original_open = builtins.open
def robust_utf8_open(file, mode='r', *args, **kwargs):
    if 'b' not in mode: kwargs.setdefault('encoding', 'utf-8')
    return original_open(file, mode, *args, **kwargs)
builtins.open = robust_utf8_open

import styletts2_inference.models
styletts2_inference.models.open = robust_utf8_open

def fake_hf_hub_download(repo_id, filename, **kwargs):
    if os.path.exists(repo_id): return os.path.join(repo_id, filename)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, **kwargs)
styletts2_inference.models.hf_hub_download = fake_hf_hub_download

# Відновлюємо вивід для власних логів
sys.stdout, sys.stderr = _original_stdout, _original_stderr

from logger_config import log
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("inputs", "temp_voice.wav")
stt_to_llm, llm_to_tts = asyncio.Queue(), asyncio.Queue()
stop_ev, inter_ev = asyncio.Event(), asyncio.Event()

def record_microphone_core(filename, sample_rate=16000, threshold=0.035, silence_duration=1.2):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    chunk_size = 1024
    audio_buffer, is_speaking = [], False
    silence_samples, max_silence_samples = 0, int((silence_duration * sample_rate) / chunk_size)
    pre_record_len, pre_record_buffer, raw_buffer = int((0.4 * sample_rate) / chunk_size), [], []
    def callback(indata, frames, time_info, status): raw_buffer.append(indata.copy())
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
            else: t.sleep(0.01)
    if audio_buffer and not stop_ev.is_set():
        sf.write(filename, np.concatenate(audio_buffer, axis=0), sample_rate)
        return True
    return False

async def input_stt_worker(text_mode, stt_module):
    while not stop_ev.is_set():
        if text_mode:
            user_input = await asyncio.to_thread(lambda: input("👤 Ви: ").strip())
            if user_input: await stt_to_llm.put(user_input)
        else:
            recorded = await asyncio.to_thread(record_microphone_core, TEMP_AUDIO_PATH)
            if recorded:
                user_input = await asyncio.to_thread(stt_module.transcribe_audio, TEMP_AUDIO_PATH)
                if user_input:
                    log.info(f"👤 Ви: {user_input}")
                    await stt_to_llm.put(user_input)

async def llm_worker(llm):
    while not stop_ev.is_set():
        user_input = await stt_to_llm.get()
        inter_ev.clear()
        for text_chunk, _ in llm.generate_response(user_input, inter_ev):
            if inter_ev.is_set(): break
            await llm_to_tts.put(text_chunk)
        stt_to_llm.task_done()

async def tts_worker(tts):
    while not stop_ev.is_set():
        sentence = await llm_to_tts.get()
        if not inter_ev.is_set():
            await asyncio.to_thread(tts.say, sentence)
        llm_to_tts.task_done()

async def main_async():
    with open(CONFIG_PATH, "r") as f: cfg = json.load(f)
    stt_m = AudioHandler(cfg)
    llm = LLMHandler(cfg)
    tts_m = TTSHandler(cfg)
    tasks = [asyncio.create_task(input_stt_worker(cfg.get('text_mode', False), stt_m)),
             asyncio.create_task(llm_worker(llm)),
             asyncio.create_task(tts_worker(tts_m))]
    log.info("🚀 [SYSTEM] Роботу конвеєра стабілізовано!")
    await stop_ev.wait()

if __name__ == "__main__": asyncio.run(main_async())