import os
import numpy as np
import onnxruntime as ort


class ONNXEngine:
    def __init__(self, onnx_path, n_threads=4):
        """
        Ініціалізує двигун ONNX з оптимізаціями для CPU (Intel i5-8265U).
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Модель не знайдена: {onnx_path}")

        # Налаштування сесії ONNX для CPU
        sess_options = ort.SessionOptions()

        # 4 потоки для i5-8265U - оптимальний баланс, що запобігає перегріву (тротлінгу)
        sess_options.intra_op_num_threads = n_threads

        # Послідовне виконання - найкращий вибір для малих тензорів StyleTTS2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Увімкнення всіх оптимізацій графа (Graph Fusion, Constant Folding)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Оптимізації пам'яті для стабільності роботи
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True

        # Вимкнення зайвих логів для економії часу
        sess_options.log_severity_level = 3

        # Ініціалізація сесії
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

    def generate(self, tokens, speed, style):
        """
        Генерує аудіо з фонем та стилю.
        """
        # 1. Підготовка токенів: конвертація в numpy та додавання технічного нуля
        tokens_np = tokens.detach().cpu().numpy() if hasattr(tokens, 'detach') else tokens
        tokens_onnx = np.concatenate([[0], tokens_np]).astype(np.int64)

        # 2. Підготовка стилю: переконаємося, що це float32 і розмірність [1, N]
        style_np = style.detach().cpu().numpy() if hasattr(style, 'detach') else style
        if style_np.ndim == 1:
            style_np = np.expand_dims(style_np, axis=0)

        # 3. Формування вхідних даних для ONNX
        inputs = {
            'tokens': tokens_onnx,
            'speed': np.array(speed, dtype=np.float32),
            's_prev': style_np.astype(np.float32)
        }

        # 4. Виконання інференсу
        # run повертає список результатів, ми беремо перший (аудіо-хвиля)
        return self.session.run(None, inputs)[0].flatten()


# --- Фабричний метод для інтеграції ---
def get_tts_engine(config_tts):
    """
    Фабрика для отримання екземпляра двигуна.
    Тепер за замовчуванням беремо спрощену модель без квантування.
    """
    onnx_path = config_tts.get("onnx_path", "models/styletts2_sim.onnx")
    n_threads = config_tts.get("n_threads", 4)
    return ONNXEngine(onnx_path=onnx_path, n_threads=n_threads)