import time, os, sys
from llama_cpp import Llama

class LLMHandler:
    def __init__(self, config):
        self.llm_config = config.get('llm', {})
        self.max_words = self.llm_config.get('min_sentence_len', 3)
        # Примусова українська мова
        self.system_prompt = config.get('character', {}).get('system_prompt', "Ти Селті. Спілкуйся виключно українською.")
        self.model = Llama(
            model_path=self.llm_config.get('model_path', ''),
            n_ctx=512, n_threads=4, verbose=False
        )

    def generate_response(self, text, interrupt_event):
        prompt = f"System: {self.system_prompt}\nUser: {text}\nAssistant:"
        stream = self.model(prompt, stream=True, max_tokens=150)
        buffer = ""
        for chunk in stream:
            if interrupt_event.is_set(): break
            token = chunk["choices"][0]["text"]
            buffer += token
            if len(buffer.split()) >= self.max_words and any(m in token for m in [".", "!", "?", ",", "\n"]):
                yield buffer.strip(), time.time()
                buffer = ""
        if buffer.strip(): yield buffer.strip(), time.time()