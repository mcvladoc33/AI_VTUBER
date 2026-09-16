import os
import sys
import re
from llama_cpp import Llama

class LLMHandler:
    def __init__(self, config):
        self.llm_config = config.get('llm', {})
        self.char_config = config.get('character', {})
        self.character_name = self.char_config.get('name', 'Помічниця')

        if not os.path.exists(self.llm_config.get('model_path', '')):
            print(f"❌ ПОМИЛКА [LLM]: Файл моделі не знайдено: {self.llm_config.get('model_path')}")
            self.model = None
            return

        print(f"🧠 [LLM] Завантаження моделі {self.character_name}...")

        sys.stdout.flush()
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)

                self.model = Llama(
                    model_path=self.llm_config['model_path'],
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

        print("✅ [LLM] Модель мислення готова.")

        # Прогрів: проганяємо системний промпт, щоб перша жива репліка
        # не платила за холодний prefill
        try:
            print("🔥 [LLM] Прогрів моделі...")
            import time
            t0 = time.time()
            self.model(
                prompt=f"System: {self.char_config.get('system_prompt', '')}\nUser: Привіт\nAssistant:",
                max_tokens=1,
                stream=False,
                echo=False
            )
            print(f"✅ [LLM] Прогрів завершено за {time.time() - t0:.2f}s.")
        except Exception as e:
            print(f"⚠️ [LLM] Прогрів не вдався (некритично): {e}")

    def generate_response(self, text: str):
        if not self.model:
            yield "Помилка: Модель ШІ не завантажена."
            return

        prompt = f"System: {self.char_config.get('system_prompt', '')}\nUser: {text}\nAssistant:"

        response_stream = self.model(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 180),
            temperature=self.llm_config.get('temperature', 0.65),
            stop=["User:", "System:", "Assistant:", "\nUser"],
            stream=True,
            echo=False
        )

        # Гібридна стратегія: перший шматок — маленький (швидкий старт),
        # решта — великі (економія на оверхеді TTS)
        FIRST_MIN = 15    # перше речення віддаємо майже одразу
        NEXT_MIN = 100    # далі накопичуємо
        NEXT_MAX = 120    # але не більше, щоб StyleTTS2 не деградував

        buf = ""
        is_first = True

        for chunk in response_stream:
            buf += chunk["choices"][0]["text"]

            if is_first:
                # ВИПРАВЛЕНО: раніше re.search() завжди знаходив найперший
                # знак пунктуації і намертво "застрягав" на ньому, якщо те
                # речення виявлялось коротшим за FIRST_MIN. Тепер перебираємо
                # всі знайдені збіги, поки не знайдемо перший достатньо довгий.
                found = False
                for m in re.finditer(r'[.!?…]+(?=\s|$)', buf):
                    candidate = buf[:m.end()].strip()
                    if len(candidate) >= FIRST_MIN:
                        buf = buf[m.end():]
                        is_first = False
                        found = True
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
                        yield part
                elif len(buf) >= NEXT_MAX:
                    sp = window.rfind(' ')
                    sp = sp if sp > 0 else NEXT_MAX
                    part = buf[:sp].strip()
                    buf = buf[sp:]
                    if part:
                        yield part

        if buf.strip():
            yield buf.strip()