import os
import sys
import re
from llama_cpp import Llama
from logger_config import log
from num2words import num2words


class LLMHandler:
    def __init__(self, config):
        self.llm_config = config.get('llm', {})
        self.char_config = config.get('character', {})
        self.character_name = self.char_config.get('name', 'Селті')

        model_path = self.llm_config.get('model_path', '')
        if not os.path.exists(model_path):
            log.critical(f"❌ [LLM] Файл моделі не знайдено за вказаним шляхом: {model_path}")
            self.model = None
            return

        log.info(f"🧠 [LLM] Завантаження GGUF файлу моделі: {model_path} на CPU...")

        sys.stdout.flush()
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)

                self.model = Llama(
                    model_path=model_path,
                    n_ctx=self.llm_config.get('n_ctx', 512),
                    n_threads=self.llm_config.get('n_threads', 4),
                    n_batch=self.llm_config.get('n_batch', 8),
                    verbose=False
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        log.info("✅ [LLM] Модель мислення успішно інтегрована.")

    def _replace_numbers_with_words(self, text: str) -> str:
        def replace(match):
            try:
                return num2words(int(match.group()), lang='uk')
            except:
                return match.group()

        return re.sub(r'\d+', replace, text)

    def generate_response(self, text: str, interrupt_event=None):
        if not self.model:
            yield "Помилка конфігурації моделі."
            return

        prompt = f"System: {self.char_config.get('system_prompt', '')}\nUser: {text}\nAssistant:"

        response_stream = self.model(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 60),
            temperature=self.llm_config.get('temperature', 0.4),
            stop=["User:", "System:", "Assistant:", "\nUser", "\n\n", "1.", "2."],
            stream=True,
            echo=False
        )

        token_buffer = ""
        for chunk in response_stream:
            if interrupt_event and interrupt_event.is_set():
                break

            token = chunk["choices"][0]["text"]
            token_buffer += token

            if re.search(r'[.!?\n]', token_buffer):
                clean_sentence = re.sub(r'<think>.*?</think>', '', token_buffer, flags=re.DOTALL).strip()
                clean_sentence = re.sub(r'<think>.*', '', clean_sentence, flags=re.DOTALL).strip()
                clean_sentence = clean_sentence.replace('"', '').replace('*', '').strip()

                clean_sentence = self._replace_numbers_with_words(clean_sentence)

                if len(clean_sentence) >= 6:
                    yield clean_sentence
                    token_buffer = ""

        if token_buffer.strip() and not (interrupt_event and interrupt_event.is_set()):
            clean_sentence = re.sub(r'<think>.*?</think>', '', token_buffer, flags=re.DOTALL).strip()
            clean_sentence = clean_sentence.replace('"', '').replace('*', '').strip()
            clean_sentence = self._replace_numbers_with_words(clean_sentence)
            if len(clean_sentence) >= 2:
                yield clean_sentence