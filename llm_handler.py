import os
import sys
import re
import time
from llama_cpp import Llama
from logger_config import log
from num2words import num2words


class LLMHandler:
    def __init__(self, config):
        self.llm_config = config.get('llm', {})
        self.char_config = config.get('character', {})
        self.history = []
        self.max_history_turns = 4

        model_path = self.llm_config.get('model_path', '')
        if not os.path.exists(model_path):
            log.critical(f"❌ [LLM] Файл моделі не знайдено: {model_path}")
            self.model = None
            return
        log.info(f"🧠 [LLM] Ініціалізація GGUF моделі з кешуванням...")

        old_stdout, old_stderr = os.dup(1), os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=self.llm_config.get('n_ctx', 512),
                    n_threads=self.llm_config.get('n_threads', 4),
                    n_threads_batch=self.llm_config.get('n_threads_batch', 4),
                    n_batch=self.llm_config.get('n_batch', 32),
                    f16_kv=True,
                    use_mmap=True,
                    embedding=False,
                    verbose=False
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        log.info("✅ [LLM] Модель інтегрована (Кеш та відображення пам'яті активні).")

    def _clean_text_for_tts(self, text: str) -> str:
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\b(пф|хм|оу|хаха|ахах|хехе|ее|е\-е)\b', '', text, flags=re.IGNORECASE)

        # Замінюємо довгі тире на коми, щоб уникнути заїкань в StyleTTS2
        text = text.replace('—', ',').replace(' – ', ', ').replace(' - ', ', ')
        text = text.replace('’', "'").replace('`', "'")

        # Дозволяємо лише базові розділові знаки та апостроф
        text = re.sub(r'[^\w\s.,!?:\u0027іІїЇєЄґҐ]', '', text)
        text = self._replace_numbers_with_words(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _replace_numbers_with_words(self, text: str) -> str:
        def replace(match):
            try:
                return num2words(int(match.group()), lang='uk')
            except:
                return match.group()

        return re.sub(r'\d+', replace, text)

    def generate_response(self, text: str, interrupt_event=None):
        if not self.model: return

        self.history.append({"role": "user", "text": text})

        prompt = f"System: {self.char_config.get('system_prompt', '')}\n"
        for turn in self.history:
            prompt += f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['text']}\n"
        prompt += "Assistant:"

        response_stream = self.model(
            prompt=prompt,
            max_tokens=self.llm_config.get('max_tokens', 400),
            temperature=self.llm_config.get('temperature', 0.6),
            repeat_penalty=self.llm_config.get('repeat_penalty', 1.2),
            stop=["User:", "System:", "Assistant:", "\nUser"],
            stream=True
        )

        full_response = ""
        token_buffer = ""
        last_sentence_time = time.time()

        for chunk in response_stream:
            if interrupt_event and interrupt_event.is_set():
                break

            token = chunk["choices"][0]["text"]
            token_buffer += token

            should_split = False
            if re.search(r'[.!?\n]', token_buffer) and (
                    token_buffer.endswith(' ') or re.search(r'[.!?\n]\s*$', token_buffer)):
                should_split = True
                # Зменшили поріг з 45 до 38 символів для швидшого стрімінгу шматків тексту через кому
            elif len(token_buffer) > 38 and ',' in token_buffer and (
                    token_buffer.endswith(' ') or token_buffer.endswith(',')):
                should_split = True

            if should_split:
                clean_sentence = self._clean_text_for_tts(token_buffer)
                clean_sentence = re.sub(r'^[.,!?—\-:\s]+', '', clean_sentence).strip()

                if len(clean_sentence) >= 5:
                    current_time = time.time()
                    gen_delta = current_time - last_sentence_time
                    last_sentence_time = current_time

                    yield clean_sentence, gen_delta
                    full_response += " " + clean_sentence
                    token_buffer = ""

                    # Фінальний хвіст
        if token_buffer.strip() and not (interrupt_event and interrupt_event.is_set()):
            clean_sentence = self._clean_text_for_tts(token_buffer)
            clean_sentence = re.sub(r'^[.,!?—\-:\s]+', '', clean_sentence).strip()

            if not any(clean_sentence.endswith(char) for char in ['.', '!', '?', ',']):
                clean_sentence = re.sub(r'\s+\w+$', '', clean_sentence).strip()

            if len(clean_sentence) >= 3:
                current_time = time.time()
                gen_delta = current_time - last_sentence_time
                yield clean_sentence, gen_delta
                full_response += " " + clean_sentence

        if full_response.strip():
            self.history.append({"role": "assistant", "text": full_response.strip()})

        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]