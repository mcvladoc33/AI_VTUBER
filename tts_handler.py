import os
import re
import sys
import warnings
import logging

# =====================================================================
# 🛠️ КРИТИЧНИЙ ПАТЧ ДЛЯ FFMPEG (Має відпрацювати до імпорту pydub)
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG_BIN = os.path.join(BASE_DIR, "bin")

if FFMPEG_BIN not in os.environ["PATH"]:
    os.environ["PATH"] = FFMPEG_BIN + os.path.pathsep + os.environ["PATH"]

from pydub import AudioSegment

AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")
# =====================================================================

import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from unicodedata import normalize
from num2words import num2words

# Імпорт наголосів з правильного підмодуля
from ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import styletts2_inference.models


def fake_hf_hub_download(repo_id, filename, **kwargs):
    local_file_path = os.path.join(repo_id, filename)
    if os.path.exists(local_file_path):
        return local_file_path
    raise FileNotFoundError(f"Файл моделі не знайдено локально: {local_file_path}")


styletts2_inference.models.hf_hub_download = fake_hf_hub_download

original_open = open


def utf8_open(*args, **kwargs):
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return original_open(*args, **kwargs)


styletts2_inference.models.open = utf8_open


class TTSHandler:
    def __init__(self, config):
        self.config = config
        self.tts_config = config.get('tts', {})

        self.styletts_path = os.path.join(BASE_DIR, "models", "styletts2_ukrainian_multispeaker")
        self.verbalizer_path = os.path.join(BASE_DIR, "models", "mbart-large-50-verbalization")
        self.preset_dir = os.path.join(BASE_DIR, "voices")
        self.ref_dir = os.path.join(BASE_DIR, "references")
        self.output_dir = os.path.join(BASE_DIR, "outputs")

        os.makedirs(self.preset_dir, exist_ok=True)
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📥 [TTS] Ініціалізація StyleTTS2 UA (Робота на: {self.device.upper()})...")

        self.mode = self.tts_config.get("mode", "1")
        self.speed = self.tts_config.get("speed", 1.0)
        self.noise_scale = self.tts_config.get("noise_scale", 0.1)
        self.match_duration = self.tts_config.get("match_duration", False)
        self.use_verbalizer = self.tts_config.get("use_verbalizer", True)

        self.verbalizer_model = None
        self.tokenizer = None
        if self.use_verbalizer:
            from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
            try:
                self.tokenizer = MBart50TokenizerFast.from_pretrained(self.verbalizer_path, local_files_only=True)
                self.verbalizer_model = MBartForConditionalGeneration.from_pretrained(self.verbalizer_path,
                                                                                      local_files_only=True).to(
                    self.device)
                self.tokenizer.src_lang = "uk_UA"
                self.tokenizer.tgt_lang = "uk_UA"
                print("✅ [TTS] Вербалізатор mBART успішно завантажено в пам'ять.")
            except Exception as e:
                print(f"⚠️ [TTS] mBART не завантажено ({e}), працює швидка алгоритмічна заміна.")

        from styletts2_inference.models import StyleTTS2
        self.multi_model = StyleTTS2(hf_path=self.styletts_path, device=self.device)

        self.stressify = Stressifier()
        self.ipa_func = ipa

        self.style = None
        self.target_duration = None
        self._prepare_voice()

    def _prepare_voice(self):
        if self.mode == "1":
            preset_file = self.tts_config.get("preset_filename", "Інна Гелевера.pt")
            preset_path = os.path.join(self.preset_dir, preset_file)
            if not os.path.exists(preset_path):
                raise FileNotFoundError(f"❌ Пресет '{preset_file}' не знайдено у папці {self.preset_dir}")
            self.style = torch.load(preset_path, map_location=self.device)
            print(f"👤 [TTS] Успішно активовано пресет голосу: {preset_file}")
        elif self.mode == "2":
            ref_file = self.tts_config.get("reference_filename", "sample.wav")
            ref_path = os.path.join(self.ref_dir, ref_file)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"❌ Референс '{ref_file}' не знайдено у папці {self.ref_dir}")

            self.style = self.multi_model.extract_voice_features(ref_path)
            if isinstance(self.style, list):
                self.style = self.style[-1]
            self.style = self.style.to(self.device)

            y, _ = librosa.load(ref_path, sr=24000)
            self.target_duration = librosa.get_duration(y=y, sr=24000)
            print(f"🎭 [TTS] Активовано клонування голосу з файлу: {ref_file}")

    def _split_to_parts(self, text_data):
        split_symbols = '.?!:'
        parts = ['']
        index = 0
        for s in text_data:
            parts[index] += s
            if s in split_symbols and len(parts[index]) > 150:
                index += 1
                parts.append('')
        return [p.strip() for p in parts if p.strip()]

    def generate_speech(self, text: str):
        if not text.strip():
            return

        try:
            # 🧹 Текстовий фільтр проти збоїв нульових тензорів та емодзі
            clean_text = text.strip()
            clean_text = re.sub(r'\.{2,}', '.', clean_text)
            clean_text = re.sub(r'[^\w\s\d.,!?;:()\-—–\'"\«\»]', '', clean_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            if not clean_text:
                return

            raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
            processed_sentences = []

            for s in raw_sentences:
                if not s.strip():
                    continue
                if self.use_verbalizer and self.verbalizer_model and re.search(r'\d+', s):
                    inputs = self.tokenizer(s, return_tensors="pt", padding=True).to(self.device)
                    generated_tokens = self.verbalizer_model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id["uk_UA"],
                        max_length=len(s) + 40,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )
                    clean_s = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    processed_sentences.append(clean_s.strip())
                else:
                    processed_sentences.append(s.strip())

            final_text = " ".join(processed_sentences)
            final_text = re.sub(r'\badvel\b', 'Адвел', final_text, flags=re.IGNORECASE)

            if re.search(r'\d+', final_text):
                final_text = re.sub(r'\d+', lambda m: num2words(int(m.group(0)), lang='uk'), final_text)

            parts = self._split_to_parts(final_text)
            final_speed = self.speed

            if self.mode == "2" and self.match_duration and self.target_duration:
                temp_wavs = []
                for t in parts:
                    t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
                    ps = self.ipa_func(self.stressify(t_norm))
                    if ps:
                        tokens = self.multi_model.tokenizer.encode(ps)
                        w = self.multi_model(tokens, speed=1.0, s_prev=self.style)
                        temp_wavs.append(w)
                if temp_wavs:
                    gen_len = sum(len(w) for w in temp_wavs) / 24000
                    calc_speed = gen_len / self.target_duration
                    if 0.6 <= calc_speed <= 1.4:
                        final_speed = calc_speed

            result_wav = []
            for t in parts:
                t_norm = normalize('NFKC', t.replace('+', StressSymbol.CombiningAcuteAccent))
                ps = self.ipa_func(self.stressify(t_norm))
                if ps:
                    tokens = self.multi_model.tokenizer.encode(ps)
                    current_style = self.style.clone()
                    if self.noise_scale > 0:
                        current_style += torch.randn_like(current_style) * self.noise_scale

                    wav = self.multi_model(tokens, speed=final_speed, s_prev=current_style)
                    result_wav.append(wav.cpu().numpy().flatten())

            if not result_wav:
                print("❌ ПОМИЛКА [TTS]: Не вдалося розпізнати фонети.")
                return

            audio_data = np.concatenate(result_wav)
            sd.play(audio_data, 24000)

            wav_output_path = os.path.join(self.output_dir, "output.wav")
            mp3_output_path = os.path.join(self.output_dir, "output.mp3")
            sf.write(wav_output_path, audio_data, 24000)

            try:
                clipped_audio = np.clip(audio_data, -1.0, 1.0)
                int16_audio = (clipped_audio * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    int16_audio.tobytes(),
                    frame_rate=24000,
                    sample_width=2,
                    channels=1
                )
                audio_segment.export(mp3_output_path, format="mp3", bitrate="192k")
            except Exception as mp3_err:
                print(f"⚠️ [TTS] Не вдалося експортувати MP3: {mp3_err}")

            sd.wait()
            print("🔊 [TTS] Програвання голосу завершено.")

        except Exception as e:
            print(f"❌ ПОМИЛКА [TTS] Критичний збій синтезу: {e}")