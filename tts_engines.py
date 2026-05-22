import os
import sys
import torch
import numpy as np
import onnxruntime as ort

class TorchEngine:
    def __init__(self, model):
        self.model = model

    def generate(self, tokens, speed, style):
        # Стандартний PyTorch інференс
        return self.model(tokens, speed=speed, s_prev=style).cpu().numpy().flatten()


class ONNXEngine:
    def __init__(self, onnx_path, n_threads=4):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Критична помилка: ONNX модель не знайдена за шляхом {onnx_path}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = n_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Ініціалізуємо швидку сесію для процесора
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=options,
            providers=['CPUExecutionProvider']
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def generate(self, tokens, speed, style):
        # Безпечне від'єднання від графа обчислень (виправлення помилок grad)
        if hasattr(tokens, 'detach'):
            tokens_np = tokens.detach().cpu().numpy()
        else:
            tokens_np = tokens

        # Додаємо технічний нуль на початок масиву фонем/токенів
        tokens_onnx = np.concatenate([[0], tokens_np]).astype(np.int64)

        # СУВОРЕ ВИПРАВЛЕННЯ: Залишаємо ОДНОМІРНИЙ масив (Expected: 1) відповідно до вимог твоєї моделі

        # Обробка пресету стилю (перетворення з Torch-тензора у Numpy)
        style_np = style.detach().cpu().numpy() if hasattr(style, 'detach') else style
        if len(style_np.shape) == 1:
            style_np = np.expand_dims(style_np, axis=0)

        # Формуємо вхідні тензори ТІЛЬКИ в одновимірному форматі для tokens та speed
        inputs = {
            'tokens': tokens_onnx,
            'speed': np.array(speed, dtype=np.float32),  # Чистий скаляр у форматі numpy без батч-вимірності
            's_prev': style_np.astype(np.float32)
        }

        # Виконуємо оптимізований С++ інференс та повертаємо одномірний аудіо-масив
        return self.session.run(None, inputs)[0].flatten()


def get_tts_engine(config_tts, pytorch_model=None):
    """
    Фабричний метод для автоматичного створення потрібного двигуна
    на основі файлу конфігурації config.json
    """
    engine_type = config_tts.get("engine", "torch").lower()

    if engine_type == "onnx":
        onnx_path = config_tts.get("onnx_path", "models/styletts2.onnx")
        n_threads = config_tts.get("n_threads", 4)
        return ONNXEngine(onnx_path=onnx_path, n_threads=n_threads)
    else:
        if pytorch_model is None:
            raise ValueError("Для використання TorchEngine необхідно передати завантажену модель PyTorch.")
        return TorchEngine(pytorch_model)