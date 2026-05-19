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
        self.history = []
        self.max_history_turns = 10

        model_path = self.llm_config.get('model_path', '')
        if not os.path.exists(model_path):
            log.critical(f"❌ [LLM] Файл моделі не знайдено: {model_path}")
            self.model = None
            return

        log.info(f"🧠 [LLM] Ініціалізація GGUF моделі...")

        old_stdout, old_stderr = os.dup(1), os.dup(2)
        try:
            with open(os.devnull, 'w') as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=self.llm_config.get('n_ctx', 2048),
                    n_threads=self.llm_config.get('n_threads', 4),
                    n_batch=self.llm_config.get('n_batch', 32),
                    f16_kv=True,
                    verbose=False
                )
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        log.info("✅ [LLM] Модель інтегрована (KV-кеш стабільний).")

    def _clean_text_for_tts(self, text: str) -> str:
        """ Очищення тексту для StyleTTS2 """
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)

        # Видаляємо звукові паразити
        text = re.sub(r'\b(пф|хм|оу|хаха|ахах|хехе|ее|е\-е)\b', '', text, flags=re.IGNORECASE)

        # Тільки українські літери, цифри та пунктуація
        text = re.sub(r'[^\w\s.,!?—\-:;іІїЇєЄґҐ\']', '', text)
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
            max_tokens=self.llm_config.get('max_tokens', 250),
            temperature=self.llm_config.get('temperature', 0.7),
            stop=["User:", "System:", "Assistant:", "\nUser"],
            stream=True
        )

        full_response = ""
        token_buffer = ""

        for chunk in response_stream:
            if interrupt_event and interrupt_event.is_set():
                break

            token = chunk["choices"][0]["text"]
            token_buffer += token

            # Перевіряємо закінчення речення (. ! ? або новий рядок)
            if re.search(r'[.!?\n]', token_buffer):
                # Захист від розриву слів: віддаємо, тільки якщо слово завершене пробілом
                if token_buffer.endswith(' ') or re.search(r'[.!?\n]\s*$', token_buffer):
                    clean_sentence = self._clean_text_for_tts(token_buffer)

                    # ГРУПУВАННЯ: Якщо назбиралося менше 35 символів (поодинокі слова),
                    # ми НЕ віддаємо їх в TTS, а чекаємо наступного токена для склеювання.
                    if len(clean_sentence) >= 35:
                        yield clean_sentence
                        full_response += " " + clean_sentence
                        token_buffer = ""

        # Вигрібаємо залишки тексту (включаючи поодинокі слова наприкінці репліки)
        if token_buffer.strip() and not (interrupt_event and interrupt_event.is_set()):
            clean_sentence = self._clean_text_for_tts(token_buffer)
            if len(clean_sentence) >= 2:
                yield clean_sentence
                full_response += " " + clean_sentence

        if full_response.strip():
            self.history.append({"role": "assistant", "text": full_response.strip()})
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]