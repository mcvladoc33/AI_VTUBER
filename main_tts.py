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
from tts_handler2 import TTSHandler

CONFIG_PATH = "config.json"
TEMP_AUDIO_PATH = os.path.join("input", "temp_voice.wav")

if __name__ == "__main__":

    # --- НОВИЙ КОД: Завантажуємо налаштування з файлу ---
    log.info(f"Завантаження налаштувань з {CONFIG_PATH}...")
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        log.error(f"❌ Файл {CONFIG_PATH} не знайдено! Будуть використані стандартні налаштування.")
        config = {}
    # ----------------------------------------------------

    # 1. Ініціалізуємо клас, тепер змінна config існує
    tts = TTSHandler(config)

    # 2. Прогріваємо модель при старті (щоб перша фраза не "гальмувала")
    tts.warmup()

    # --- Коли прийшов час говорити репліку ---

    # 3. Відкриваємо сесію (очищуємо буфер)
    tts.start_session()

    # 4. Передаємо шматки тексту по одному. 
    # Зауваж: текст ВЖЕ має бути нарізаний модулем SentenceBuffer!
    chunk_1 = "Привіт, як твої справи?"
    tts.synthesize_and_play(chunk_1) # Згенерує і дочекається кінця відтворення

    chunk_2 = "Сьогодні чудова погода."
    tts.synthesize_and_play(chunk_2) # Згенерує і дочекається кінця відтворення

    # 5. Завершуємо сесію. Це автоматично склеїть усе в один файл output.wav
    tts.finish_session()

    # 6. Коли вимикаєш програму
    tts.close()