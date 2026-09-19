import os
import re
import time
import ctypes
from llama_cpp import Llama
import llama_cpp
from logger_config import log


# --- Фікс "Exception ignored on calling ctypes callback function" ---
# llama.cpp логує через C-колбек (llama_log_set). Якщо не тримати сильне
# посилання на цей колбек десь на рівні модуля, Python GC прибирає його
# ще до того, як бібліотека встигає ним скористатись — звідси
# "Exception ignored ... llama_log_callback" в консолі. verbose=False у
# Llama(...) не рятує — він лише притишує python-рівень, а не C-рівень.
# Реєструємо власний мовчазний колбек і тримаємо сильне посилання на
# module-рівні (_LLAMA_LOG_CALLBACK), щоб GC його не чіпав — це заразом
# і прибирає шумні рядки типу "n_ctx_seq (...) < n_ctx_train (...)".
def _silent_llama_log(level, message, user_data):
    pass


_LLAMA_LOG_CALLBACK = llama_cpp.llama_log_callback(_silent_llama_log)
llama_cpp.llama_log_set(_LLAMA_LOG_CALLBACK, ctypes.c_void_p())


class LLMHandler:
    def __init__(self, config):
        self.llm_config = config.get('llm', {})
        self.char_config = config.get('character', {})
        self.character_name = self.char_config.get('name', 'Помічниця')

        self.history = []
        self.max_history_turns = 4

        model_path = self.llm_config.get('model_path', '')
        if not os.path.exists(model_path):
            log.critical(f"❌ [LLM] Файл моделі не знайдено: {model_path}")
            self.model = None
            return

        log.info(f"🧠 [LLM] Завантаження моделі {self.character_name}...")

        old_stdout, old_stderr = os.dup(1), os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=self.llm_config.get('n_ctx', 1024),
                    # n_threads керує decode (генерацією токенів) — тримаємо малим,
                    # щоб лишити ядра для паралельного TTS-рендеру
                    n_threads=self.llm_config.get('n_threads', 2),
                    # n_threads_batch керує prefill (обробкою промпту) — тут можна
                    # брати максимум, бо TTS в цей момент ще нічого не робить
                    n_threads_batch=self.llm_config.get('n_threads_batch', 4),
                    n_batch=self.llm_config.get('n_batch', 256),
                    flash_attn=True,
                    swa_full=False,
                    verbose=False
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        log.info("✅ [LLM] Модель мислення готова.")

        # Прогрів: проганяємо системний промпт, щоб перша жива репліка
        # не платила за холодний prefill
        try:
            log.info("🔥 [LLM] Прогрів моделі...")
            t0 = time.time()
            self.model(
                prompt=f"System: {self.char_config.get('system_prompt', '')}\nUser: Привіт\nAssistant:",
                max_tokens=1,
                stream=False,
                echo=False
            )
            log.info(f"✅ [LLM] Прогрів завершено за {time.time() - t0:.2f}s.")
        except Exception as e:
            log.warning(f"⚠️ [LLM] Прогрів не вдався (некритично): {e}")

    def generate_response(self, text: str, seed: int = None):
        if not self.model:
            yield "Помилка: Модель ШІ не завантажена."
            return

        self.history.append({"role": "user", "text": text})

        prompt = f"System: {self.char_config.get('system_prompt', '')}\n"
        for turn in self.history:
            prompt += f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['text']}\n"
        prompt += "Assistant:"

        gen_kwargs = dict(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 180),
            temperature=self.llm_config.get('temperature', 0.65),
            repeat_penalty=self.llm_config.get('repeat_penalty', 1.15),
            stop=["User:", "System:", "Assistant:", "\nUser"],
            stream=True,
            echo=False
        )
        # seed=None (дефолт) — звичайна жива розмова, щоразу інша відповідь.
        # Фіксований seed — лише для benchmark.py: та сама фраза + той самий
        # seed = той самий текст щоразу, інакше порівняння конфігурацій
        # (n_threads, engine, parallel_chunks) забруднюється ще й
        # випадковістю самого семплінгу LLM, а не лише швидкістю заліза.
        if seed is not None:
            gen_kwargs["seed"] = seed

        response_stream = self.model(**gen_kwargs)

        # Гібридна стратегія: перший шматок — маленький (швидкий старт),
        # решта — великі (економія на фіксованому оверхеді TTS)
        FIRST_MIN = 15
        NEXT_MIN = 100
        NEXT_MAX = 120

        buf = ""
        is_first = True
        full_response = ""

        for chunk in response_stream:
            buf += chunk["choices"][0]["text"]

            if is_first:
                found = False
                for m in re.finditer(r'[.!?…]+(?=\s|$)', buf):
                    candidate = buf[:m.end()].strip()
                    if len(candidate) >= FIRST_MIN:
                        buf = buf[m.end():]
                        is_first = False
                        found = True
                        full_response += " " + candidate
                        yield candidate
                        break
                if found:
                    continue
                continue

            if len(buf) >= NEXT_MIN:
                window = buf[:NEXT_MAX]
                cuts = list(re.finditer(r'[.!?…]+(?=\s|$)', window))
                if cuts:
                    end = cuts[-1].end()
                    part = buf[:end].strip()
                    buf = buf[end:]
                    if part:
                        full_response += " " + part
                        yield part
                elif len(buf) >= NEXT_MAX:
                    sp = window.rfind(' ')
                    sp = sp if sp > 0 else NEXT_MAX
                    part = buf[:sp].strip()
                    buf = buf[sp:]
                    if part:
                        full_response += " " + part
                        yield part

        if buf.strip():
            full_response += " " + buf.strip()
            yield buf.strip()

        if full_response.strip():
            self.history.append({"role": "assistant", "text": full_response.strip()})
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]