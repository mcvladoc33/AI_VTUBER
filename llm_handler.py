import os
from llama_cpp import Llama


class LLMHandler:
    def __init__(self, config):
        llm_config = config['llm']
        self.char_config = config['character']

        # Перевіряємо, чи взагалі існує файл моделі за вказаним шляхом
        if not os.path.exists(llm_config['model_path']):
            print(f"❌ ПОМИЛКА [LLM]: Файл моделі не знайдено за шляхом: {llm_config['model_path']}")
            print("Будь ласка, перевір назву файлу та папку models!")
            self.model = None
            return

        print(f"🧠 [LLM] Завантаження моделі {self.char_config['name']} (Gemma) на CPU...")

        # Ініціалізація локальної LLM через llama.cpp
        self.model = Llama(
            model_path=llm_config['model_path'],
            n_ctx=llm_config.get('n_ctx', 1024),
            n_threads=llm_config.get('n_threads', 4),  # Використовує 4 ядра твого i5
            verbose=False  # Повністю вимикає технічний спам у консолі
        )
        print("✅ [LLM] Модель мислення успішно завантажена в оперативну пам'ять.")

    def generate_response(self, text: str) -> str:
        """Приймає текст користувача і повертає зухвалу відповідь від імені Помічниці"""
        if not self.model:
            return "Помилка: Модель ШІ не завантажена."

        # Формуємо промпт за шаблоном, який найкраще розуміє ця модель
        prompt = f"System: {self.char_config['system_prompt']}\nUser: {text}\nAssistant:"

        # Запуск генерації тексту на CPU
        response = self.model(
            prompt,
            max_tokens=60,  # Короткі відповіді генеруються значно швидше
            temperature=0.8,  # Додає помічниці характеру та оригінальності
            stop=["User:", "\n", "System:"],  # Маркери зупинки, щоб модель вчасно замовкала
            echo=False
        )

        # Забираємо чистий текст відповіді та прибираємо зайві пробіли на початку/в кінці
        answer = response["choices"][0]["text"].strip()
        return answer