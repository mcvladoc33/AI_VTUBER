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

        sys.stdout.flush()
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)

                self.model = Llama(
                    model_path=self.llm_config['model_path'],
                    n_ctx=self.llm_config.get('n_ctx', 512),
                    n_threads=4,
                    n_batch=self.llm_config.get('n_batch', 16),
                    verbose=False
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        print("✅ [LLM] Модель мислення готова.")

    def generate_response(self, text: str):
        if not self.model:
            yield "Помилка: Модель ШІ не завантажена."
            return

        prompt = f"System: {self.char_config.get('system_prompt', '')}\nUser: {text}\nAssistant:"

        response_stream = self.model(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 150),
            temperature=self.llm_config.get('temperature', 0.7),
            stop=["User:", "System:", "Assistant:", "\nUser"],
            stream=True,
            echo=False
        )

        token_buffer = ""
        for chunk in response_stream:
            token = chunk["choices"][0]["text"]
            token_buffer += token

            # Віддаємо текст ТІЛЬКИ коли є повноцінне завершене речення
            if any(c in token for c in ['.', '!', '?', '\n']) and len(token_buffer.strip()) > 20:
                yield token_buffer.strip()
                token_buffer = ""

        if token_buffer.strip():
            yield token_buffer.strip()