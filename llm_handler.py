import os
import sys
import re
import time
from llama_cpp import Llama


class LLMHandler:
    def __init__(self, config):
        # Отримуємо секції з конфігу
        self.llm_config = config.get('llm', {})
        self.char_config = config.get('character', {})

        # Динамічно підтягуємо ліміт слів для нарізки чанків з min_sentence_len конфігу
        self.max_words_per_chunk = self.llm_config.get('min_sentence_len', 4)

        # Шлях до моделі
        model_path = self.llm_config.get('model_path', '')

        # Завантажуємо системний промпт характеру Селті
        self.system_prompt = self.char_config.get(
            'system_prompt',
            "Ти Селті, харизматична дівчина-стрімер. Спілкуйся живою українською мовою."
        )

        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            # Ініціалізуємо модель з параметрами з config.json
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.llm_config.get('n_ctx', 512),
                n_threads=self.llm_config.get('n_threads', 4),
                n_threads_batch=self.llm_config.get('n_threads_batch', 4),
                n_batch=self.llm_config.get('n_batch', 64),
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
        # Додаємо в промпт інструкцію про "стиль стрімера"
        prompt = (
            f"System: {self.system_prompt} "
            f"Твій стиль мовлення: дружній, впевнений, як у ведучої стріму. "
            f"Використовуй влучні порівняння та логічні зв'язки (наприклад: 'це важливо, тому що', 'цікавий момент у тому, що'). "
            f"Будь конкретною, уникай води.\n"
            f"User: {text}\n"
            f"Assistant:"
        )

        # Стрімінг токенів з використанням гнучких параметрів конфігу
        stream = self.model(
            prompt=prompt,
            stream=True,
            max_tokens=self.llm_config.get('max_tokens', 150),
            temperature=self.llm_config.get('temperature', 0.5),
            repeat_penalty=self.llm_config.get('repeat_penalty', 1.1),
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

            # Спрацьовує, як тільки набрали ліміт слів із конфігу та знайшли роздільник
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