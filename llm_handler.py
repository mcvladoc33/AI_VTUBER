import os
import sys
from llama_cpp import Llama


class LLMHandler:
    def __init__(self, config):
        self.llm_config = config.get('llm', {})
        self.char_config = config.get('character', {})

        if not os.path.exists(self.llm_config.get('model_path', '')):
            print(f"❌ ПОМИЛКА [LLM]: Файл моделі не знайдено: {self.llm_config.get('model_path')}")
            self.model = None
            return

        print(f"🧠 [LLM] Завантаження моделі {self.char_config.get('name', 'Помічниця')}...")

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
                    n_ctx=self.llm_config.get('n_ctx', 1024),
                    n_threads=self.llm_config.get('n_threads', 4),
                    verbose=False  # Вимикає базовий verbose
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        print("✅ [LLM] Модель мислення готова.")

    def generate_response(self, text: str) -> str:
        if not self.model:
            return "Помилка: Модель ШІ не завантажена."

        prompt = f"System: {self.char_config.get('system_prompt', '')}\nUser: {text}\nAssistant:"

        response = self.model(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 60),
            temperature=self.llm_config.get('temperature', 0.8),
            stop=["User:", "\n", "System:"],
            echo=False
        )

        return response["choices"][0]["text"].strip()