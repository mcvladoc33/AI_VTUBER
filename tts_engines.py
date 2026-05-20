import torch
import numpy as np
import onnxruntime as ort


class TorchEngine:
    def __init__(self, model):
        self.model = model

    def generate(self, tokens, speed, style):
        # PyTorch очікує тензори та повертає їх
        return self.model(tokens, speed=speed, s_prev=style).cpu().numpy().flatten()


class ONNXEngine:
    def __init__(self, onnx_path, n_threads):
        options = ort.SessionOptions()
        options.intra_op_num_threads = n_threads
        self.session = ort.InferenceSession(onnx_path, sess_options=options)

    def generate(self, tokens, speed, style):
        # Безпечне від'єднання від графа обчислень (виправлення помилки grad)
        if hasattr(tokens, 'detach'):
            tokens_np = tokens.detach().cpu().numpy()
        else:
            tokens_np = tokens

        # Додаємо нуль на початок (якщо потрібно для моделі)
        tokens_onnx = np.concatenate([[0], tokens_np]).astype(np.int64)

        # Обробка стилю (безпечне перетворення)
        style_np = style.detach().cpu().numpy() if hasattr(style, 'detach') else style
        if len(style_np.shape) == 1:
            style_np = np.expand_dims(style_np, axis=0)

        inputs = {
            'tokens': tokens_onnx,
            'speed': np.array(speed, dtype=np.float32),
            's_prev': style_np.astype(np.float32)
        }
        return self.session.run(None, inputs)[0].flatten()