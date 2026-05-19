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

        self.history = []
        self.max_history_turns = 6

        model_path = self.llm_config.get('model_path', '')
        if not os.path.exists(model_path):
            log.critical(f"❌ [LLM] Файл моделі не знайдено: {model_path}")
            self.model = None
            return

        log.info(f"🧠 [LLM] Завантаження GGUF моделі з KV-кешуванням на CPU...")

        sys.stdout.flush()
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)

                self.model = Llama(
                    model_path=model_path,
                    n_ctx=self.llm_config.get('n_ctx', 1024),
                    n_threads=self.llm_config.get('n_threads', 4),
                    n_batch=self.llm_config.get('n_batch', 32),
                    f16_kv=True,
                    logits_all=False,
                    verbose=False
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        log.info("✅ [LLM] Модель інтегрована з підтримкою оперативної пам'яті.")

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

        self.history.append({"role": "user", "text": text})

        prompt = f"System: {self.char_config.get('system_prompt', '')}\n"
        for turn in self.history:
            if turn["role"] == "user":
                prompt += f"User: {turn['text']}\n"
            else:
                prompt += f"Assistant: {turn['text']}\n"
        prompt += "Assistant:"

        response_stream = self.model(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 100),
            temperature=self.llm_config.get('temperature', 0.7),
            stop=["User:", "System:", "Assistant:", "\nUser", "\n\n", "1.", "2."],
            stream=True,
            echo=False
        )

        full_response = ""
        token_buffer = ""
        accumulated_sentence = ""

        for chunk in response_stream:
            if interrupt_event and interrupt_event.is_set():
                break

            token = chunk["choices"][0]["text"]
            token_buffer += token

            # Перевіряємо закінчення логічного блоку (знаки пунктуації)
            if re.search(r'[.!?\n]', token_buffer):
                clean_block = re.sub(r'<think>.*?</think>', '', token_buffer, flags=re.DOTALL).strip()
                clean_block = re.sub(r'<think>.*', '', clean_block, flags=re.DOTALL).strip()
                clean_block = clean_block.replace('"', '').replace('*', '').strip()
                clean_block = self._replace_numbers_with_words(clean_block)

                if clean_block:
                    accumulated_sentence += " " + clean_block
                    token_buffer = ""

                    # 🔥 СТРАТЕДІЯ БУФЕРИЗАЦІЇ: якщо шматок надто малий (огризок слова або < 25 символів),
                    # ми не віддаємо його в TTS відразу, а накопичуємо нормальну фразу
                    if len(accumulated_sentence.strip()) >= 25 or any(c in accumulated_sentence for c in "!?"):
                        out_text = accumulated_sentence.strip()
                        yield out_text
                        full_response += " " + out_text
                        accumulated_sentence = ""

        # Випльовуємо залишки після завершення стріму
        if token_buffer.strip() and not (interrupt_event and interrupt_event.is_set()):
            clean_block = token_buffer.replace('"', '').replace('*', '').strip()
            clean_block = self._replace_numbers_with_words(clean_block)
            if clean_block:
                accumulated_sentence += " " + clean_block

        if accumulated_sentence.strip() and not (interrupt_event and interrupt_event.is_set()):
            out_text = accumulated_sentence.strip()
            yield out_text
            full_response += " " + out_text

        full_response = full_response.strip()
        if full_response:
            self.history.append({"role": "assistant", "text": full_response})

        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]