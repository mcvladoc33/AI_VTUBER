import os
import sys
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

        # Повністю глушимо вивід C++ логів llama.cpp у консоль Windows
        sys.stdout.flush()
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)

                self.model = Llama(
                    model_path=self.llm_config['model_path'],
                    n_ctx=384,  # Мінімальний контекст для максимальної швидкості CPU
                    n_threads=4,  # Використовуємо 4 фізичні ядра процесора
                    n_batch=8,  # Маленький батч полегшує потокову генерацію
                    verbose=False  # Вимикає базовий verbose логгер
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        print("✅ [LLM] Модель мислення готова.")

    def generate_response(self, text: str):
        """
        Генерує відповідь у режимі стрімінгу. Накопичує короткі вигуки
        (менше 40 символів), щоб мова не була рваною, і віддає повноцінні речення.
        """
        if not self.model:
            yield "Помилка: Модель ШІ не завантажена."
            return

        prompt = f"System: {self.char_config.get('system_prompt', '')}\nUser: {text}\nAssistant:"

        response_stream = self.model(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 150),
            temperature=0.8,
            stop=["User:", "System:"],
            stream=True,  # Вмикаємо потокову віддачу токенів
            echo=False
        )

        sentence_buffer = ""
        sentence_endings = {'.', '!', '?', '\n'}

        for chunk in response_stream:
            token = chunk["choices"][0]["text"]
            sentence_buffer += token

            # Якщо знайшли розділовий знак в поточному токені
            if any(char in token for char in sentence_endings):
                clean_buffer = sentence_buffer.strip()

                # Перевіряємо, чи в буфері є хоча б 40 символів і чи є там літери/цифри
                # (щоб ігнорувати порожні смайли чи набори розділових знаків)
                if len(clean_buffer) >= 40 and any(c.isalnum() for c in clean_buffer):
                    yield clean_buffer
                    sentence_buffer = ""

        # Віддаємо фінальний залишок тексту, тільки якщо там є реальний текст (букви/цифри)
        final_clean = sentence_buffer.strip()
        if final_clean and any(c.isalnum() for c in final_clean):
            yield final_clean