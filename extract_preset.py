"""
Офлайн-утиліта: витягує вектор стилю голосу з референс-аудіо і зберігає
його як .pt пресет у voices/.

Навіщо це окремий скрипт, а не частина main.py:
tts.engine="onnx" не вміє клонувати голос наживо (немає voice-encoder'а
в графі), а tts.engine="pytorch" з mode=2 — вміє, але повільніший.
Цей скрипт одноразово вантажить повну PyTorch-модель, робить клонування
один раз офлайн, і після цього ти вже назавжди працюєш зі швидким ONNX +
готовим пресетом (mode=1). Повну модель більше не чіпаєш.

Використання:
    python extract_preset.py --audio references/sample.wav --name "Мій голос"
"""
import os
import sys
import argparse
import builtins

# Офлайн-режим — той самий патерн, що й у tts_handler.py
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLETTS_PATH = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
REF_DIR = os.path.join(BASE_DIR, "references")
VOICE_DIR = os.path.join(BASE_DIR, "voices")

# --- Патч кирилиці для шляхів на Windows (той самий, що і в tts_handler.py) ---
original_open = builtins.open


def utf8_open(*args, **kwargs):
    if len(args) > 1 and 'b' in args[1]:
        return original_open(*args, **kwargs)
    if 'mode' in kwargs and 'b' in kwargs['mode']:
        return original_open(*args, **kwargs)
    kwargs['encoding'] = kwargs.get('encoding', 'utf-8')
    return original_open(*args, **kwargs)


builtins.open = utf8_open

import torch
import styletts2_inference.models


# --- Той самий фейковий hf_hub_download, що й у tts_handler.py, ---
# --- щоб не тягнути мережу і не мати розсинхрону поведінки між скриптами ---
def fake_hf_hub_download(repo_id, filename, **kwargs):
    local_file_path = os.path.join(repo_id, filename)
    if os.path.exists(local_file_path):
        return local_file_path
    raise FileNotFoundError(f"Файл моделі не знайдено локально: {local_file_path}")


styletts2_inference.models.hf_hub_download = fake_hf_hub_download
styletts2_inference.models.open = utf8_open


def resolve_audio_path(raw: str) -> str:
    """
    Приймає те, що дав користувач у --audio, і повертає реальний шлях.
    Порядок спроб:
    1. Абсолютний шлях — беремо як є.
    2. Шлях, що вже існує відносно поточної робочої директорії
       (покриває "references/sample.wav", "references\\sample.wav",
       "../інша_тека/file.wav" тощо) — беремо як є, без додавання REF_DIR.
    3. Інакше вважаємо, що дали просто ім'я файлу, і шукаємо його в references/.
    """
    if os.path.isabs(raw):
        return raw
    if os.path.exists(raw):
        return os.path.abspath(raw)
    return os.path.join(REF_DIR, raw)


def main():
    parser = argparse.ArgumentParser(description="Витягти .pt пресет голосу з референс-аудіо")
    parser.add_argument(
        "--audio", default="sample.wav",
        help="Ім'я файлу в references/ (наприклад sample.wav), або будь-який шлях до .wav"
    )
    parser.add_argument(
        "--name", default=None,
        help="Назва пресету без .pt (за замовчуванням — ім'я аудіофайлу)"
    )
    args = parser.parse_args()

    audio_path = resolve_audio_path(args.audio)
    if not os.path.exists(audio_path):
        print(f"❌ Референс-аудіо не знайдено: {audio_path}")
        sys.exit(1)

    preset_name = args.name or os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(VOICE_DIR, f"{preset_name}.pt")
    os.makedirs(VOICE_DIR, exist_ok=True)

    from styletts2_inference.models import StyleTTS2

    print("⏳ Ініціалізація повної PyTorch-моделі StyleTTS2 (лише для цього одноразового кроку)...")
    model = StyleTTS2(hf_path=STYLETTS_PATH, device="cpu")

    print(f"🎙️ Вилучення вектора стилю з: {audio_path}")
    with torch.no_grad():
        style_vector = model.extract_voice_features(audio_path)
        if isinstance(style_vector, list):
            style_vector = style_vector[-1]
        torch.save(style_vector.cpu(), output_path)

    print(f"✅ Пресет збережено: {output_path}")
    print(f"👉 Тепер вкажи в config.json: \"tts.preset_filename\": \"{preset_name}.pt\", \"tts.mode\": \"1\"")
    print("   і можеш спокійно тримати \"tts.engine\": \"onnx\" для швидкого синтезу цим голосом.")


if __name__ == "__main__":
    main()
