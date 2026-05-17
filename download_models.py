import os
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

print("⏳ Початок завантаження моделей для StyleTTS2 UA...")
print("Файли завантажуються безпосередньо в папку проєкту на диск D.")

# 1. Завантаження моделі голосу StyleTTS2
print("\n📦 Завантаження ваг голосу StyleTTS2 (~540 МБ)...")
snapshot_download(
    repo_id="patriotyk/styletts2_ukrainian_multispeaker",
    local_dir=os.path.join(MODELS_DIR, "styletts2_ukrainian_multispeaker"),
    local_dir_use_symlinks=False
)

# 2. Завантаження лінгвістичної моделі mBART
print("\n📦 Завантаження вербалізатора чисел mBART-large-50 (~2.4 ГБ)...")
snapshot_download(
    repo_id="skypro1111/mbart-large-50-verbalization",
    local_dir=os.path.join(MODELS_DIR, "mbart-large-50-verbalization"),
    local_dir_use_symlinks=False
)

print("\n🎉 Усі ШІ-моделі успішно завантажено локально!")