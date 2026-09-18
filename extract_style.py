import builtins
import torch
import huggingface_hub

# 1. Патчимо вбудований open
original_open = builtins.open

def utf8_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'b' not in mode:
        kwargs['encoding'] = kwargs.get('encoding', 'utf-8')
    elif 'encoding' in kwargs:
        del kwargs['encoding']
    return original_open(*args, **kwargs)

builtins.open = utf8_open

# 2. Перехоплюємо запити до інтернету і віддаємо локальні шляхи
original_download = huggingface_hub.hf_hub_download

def local_hub_download(repo_id, filename, *args, **kwargs):
    if filename == "config.yml":
        return "models/styletts2_ukrainian_multispeaker/config.yml"
    if filename == "pytorch_model.bin":
        return "models/styletts2_ukrainian_multispeaker/pytorch_model.bin"
    return original_download(repo_id, filename, *args, **kwargs)

huggingface_hub.hf_hub_download = local_hub_download

# 3. Імпортуємо модель
from styletts2_inference.models import StyleTTS2

audio_path = "./references/fv_anime-girl-shy.mp3"
output_pt_path = "./voices/fv_anime-girl-shy.pt"

print("⏳ Ініціалізація локальної моделі...")
model = StyleTTS2(hf_path="local_override") 

print("🎙️ Вилучення вектора стилю...")
with torch.no_grad():
    # Використовуємо правильний метод!
    style_vector = model.extract_voice_features(audio_path)
    
    # Якщо функція повернула список (як у вашому основному коді), беремо останній
    if isinstance(style_vector, list):
        style_vector = style_vector[-1]
        
    torch.save(style_vector.cpu(), output_pt_path)

print(f"✅ Вектор стилю успішно збережено у {output_pt_path}!")