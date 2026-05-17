class LLMHandler:
    def __init__(self, config):
        print("🧠 [LLM] Модуль мислення готовий.")

    def generate_response(self, text: str) -> str:
        return f"Відповідь на: {text}"