import os
import sys
import re
import time
from llama_cpp import Llama


class LLMHandler:
    def __init__(self, config):
        # Робимо шматочки ще меншими, щоб розвантажити слабкий CPU
        self.max_words_per_chunk = 4
        model_path = config.get('llm', {}).get('model_path', '')

        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=512,
                n_threads=4,
                verbose=False
            )
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr

    def _clean(self, text):
        # Вирізаємо цифри, списки та англійські вставки-баги всередині слів
        text = re.sub(r'^\d+[\s.)\-]+', '', text.strip())
        text = re.sub(r'\s+\d+[\s.)\-]+', ' ', text)
        text = re.sub(r'\(.*?\)|\[.*?\]|\*+', '', text)
        text = re.sub(r'[a-zA-Z]', '', text)  # Жорстко чистимо англійські галюцинації типу "est"
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s.,!?:\u0027іІїЇєЄґҐ-]', '', text)).strip()

    def generate_response(self, text, interrupt_event=None):
        prompt = (
            f"System: Ти Селті, дівчина-стрімер. Говори від жіночого роду. "
            f"Спілкуйся живою українською мовою. "
            f"Пиши дуже просто, без складних зворотів, без цифр і без списків.\n"
            f"User: {text}\n"
            f"Assistant:"
        )

        stream = self.model(
            prompt=prompt,
            stream=True,
            max_tokens=150,  # Обмежуємо загальну довжину відповіді, щоб не перевантажувати чергу
            temperature=0.5,  # Менша температура — швидша та чіткіша генерація токенів
            stop=["User:", "<|im_end|>"]
        )

        buffer = ""
        last_time = time.time()

        split_markers = [".", "!", "?", ",", ";", "\n"]

        for chunk in stream:
            if interrupt_event and interrupt_event.is_set():
                break

            token = chunk["choices"][0]["text"]
            buffer += token

            words_count = len(buffer.split())

            # Стріляємо миттєво, як тільки є хоча б 4 слова та будь-який роздільник/пробіл
            if words_count >= self.max_words_per_chunk and (any(m in token for m in split_markers) or token.isspace()):
                if not re.search(r'\b(ст|ч|л|мл|г|кг|хв|шт)\.$', buffer.strip(), re.IGNORECASE):
                    clean_chunk = self._clean(buffer)
                    if clean_chunk and len(clean_chunk) > 2:
                        yield clean_chunk, time.time() - last_time
                        buffer = ""
                        last_time = time.time()

        if buffer.strip() and not (interrupt_event and interrupt_event.is_set()):
            clean_chunk = self._clean(buffer)
            if clean_chunk and len(clean_chunk) > 2:
                yield clean_chunk, time.time() - last_time