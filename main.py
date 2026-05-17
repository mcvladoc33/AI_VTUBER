import json
import sys
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler
from vrm_handler import VRMHandler


def load_config():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    config = load_config()

    # Створюємо екземпляри кожного класу
    audio_module = AudioHandler(config)
    llm_module = LLMHandler(config)
    tts_module = TTSHandler(config)
    vrm_module = VRMHandler(config)

    print("\n🚀 Локальна помічниця повністю запущена! Очікую голос...")

    try:
        while True:
            # 1. Запис та розпізнавання голосу за сигналом VAD
            user_text = audio_module.listen_to_text()

            if user_text:
                print(f"🗨️ Ти сказав: {user_text}")

                # 2. Обробка тексту штучним інтелектом
                ai_response = llm_module.generate_response(user_text)
                print(f"🤖 Помічниця сформувала відповідь: {ai_response}")

                # 3. Синхронізація губ (LipSync) та емоції аватара
                vrm_module.send_to_avatar(ai_response, emotion="sarcastic")

                # 4. Фізичне відтворення звуку мовлення
                tts_module.generate_speech(ai_response)

            else:
                print("🤷 Звук був занадто тихий або слів не розпізнано. Спробуй ще раз.")

    except KeyboardInterrupt:
        print("\n👋 Програму зупинено користувачем. Бувай!")
        sys.exit(0)


if __name__ == "__main__":
    main()