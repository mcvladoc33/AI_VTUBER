"""
benchmark.py — автоматизований бенчмарк пайплайна LLM+TTS на фіксованій
фразі, на ТВОЄМУ конкретному залізі.

Навіщо: замість вручну перезапускати main.py по колу з різними
config.json і переписувати цифри з логів (як ми робили кілька разів
підряд у чаті) — цей скрипт сам перебирає список конфігурацій, кожну
по кілька разів (бо одиничний замір шумний, ±10-20%), і виводить
усереднену порівняльну таблицю + CSV для подальшого аналізу.

Мікрофон і STT тут НЕ беруть участі — фраза подається напряму в LLM,
так само, як у text_mode. Це навмисно: мікрофон вносить свою власну
мінливість (гучність голосу, фонові шуми), яка забила б порівняння
конфігурацій шумом, що не має стосунку до самого пайплайна.

Використання:
    python benchmark.py
    python benchmark.py --reps 5
    python benchmark.py --phrase "Привіт, як справи?"
    python benchmark.py --mute
    python benchmark.py --csv results.csv

--mute вимикає РЕАЛЬНЕ відтворення звуку (підміняє sd.play/sd.wait на
no-op) — проходить варіанти сильно швидше, але тоді "тиша між шматками"
перестає включати реальну тривалість відтворення попереднього шматка і
показує лише розрив між завершеннями синтезу. Для фінальної перевірки
найкращого варіанта краще прогнати ще раз БЕЗ --mute.

Список варіантів для порівняння — змінна VARIANTS нижче. Редагуй її
під те, що саме хочеш порівняти; кожен варіант — це часткові
перекриття (override) поверх твого поточного config.json.
"""
import os
import sys
import csv
import copy
import json
import time
import gc
import argparse
import statistics
from datetime import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from logger_config import log
from llm_handler import LLMHandler
from tts_handler import TTSHandler

CONFIG_PATH = "config.json"
DEFAULT_PHRASE = "привіт, як справи?"

# --- Що порівнюємо. Додавай/видаляй/редагуй варіанти тут ---------------
# Кожен варіант — (назва, overrides). overrides мерджиться поверх
# базового config.json по секціях "llm"/"tts" (глибокий мердж по ключу).
VARIANTS = [
    ("onnx / llm=1 / tts=3 / par=1",
     {"llm": {"n_threads": 1}, "tts": {"engine": "onnx", "n_threads": 3, "parallel_chunks": 1}}),

    ("onnx / llm=2 / tts=3 / par=1",
     {"llm": {"n_threads": 2}, "tts": {"engine": "onnx", "n_threads": 3, "parallel_chunks": 1}}),

    ("onnx / llm=3 / tts=3 / par=1",
     {"llm": {"n_threads": 3}, "tts": {"engine": "onnx", "n_threads": 3, "parallel_chunks": 1}}),

    # par=2 прибрано зі списку варіантів — ПІДТВЕРДЖЕНО, що ламає звук
    # (чутне заїкання на початку слів), незалежно від рушія. Не тестуємо
    # це знову, поки хтось не виправить потокобезпеку препроцесингу
    # (Stressifier/ipa_uk/StyleTTS2Tokenizer) — див. коментар у
    # tts_handler.py біля self.parallel_chunks.

    ("onnx / llm=4 / tts=4 / par=1",
     {"llm": {"n_threads": 4}, "tts": {"engine": "onnx", "n_threads": 4, "parallel_chunks": 1}}),

    ("onnx / llm=3 / tts=4 / par=1",
     {"llm": {"n_threads": 3}, "tts": {"engine": "onnx", "n_threads": 4, "parallel_chunks": 1}}),

    # pytorch — НАВМИСНО останній у списку. Повна PyTorch-модель важить
    # помітно більше (~1 ГБ понад ONNX, за раніше виміряним), і Python
    # неохоче повертає звільнену пам'ять ОС одразу — якщо цей варіант
    # запустити посередині списку, наступні (легші) варіанти можуть
    # зачепити своп і показати спотворені, повільніші числа, які насправді
    # ніяк не пов'язані з їхньою власною конфігурацією. В кінці списку —
    # шкодити вже нікому.
    ("pytorch / llm=2 / tts=3 / par=1",
     {"llm": {"n_threads": 2}, "tts": {"engine": "pytorch", "n_threads": 3, "parallel_chunks": 1}}),
]


def deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def mute_playback():
    """Підміняє sd.play/sd.wait на no-op — див. попередження в шапці
    файлу щодо впливу на точність метрики "тиші"."""
    import sounddevice as sd
    sd.play = lambda *a, **k: None
    sd.wait = lambda *a, **k: None


def run_once(llm_module: LLMHandler, tts_module: TTSHandler, phrase: str, seed: int = None) -> dict:
    """Один прогін фіксованої фрази через LLM+TTS. Повторює той самий
    інструментований цикл, що й main.py, але збирає числа в словник
    замість (чи на додачу до) друку в лог."""
    tts_module.reset_session()

    is_first = True
    start_llm = time.time()
    tts_module.start_turn_timer(start_llm)
    last_yield = start_llm
    first_token_time = None
    segment_gaps = []

    for sentence in llm_module.generate_response(phrase, seed=seed):
        now = time.time()
        if is_first:
            first_token_time = now - start_llm
            is_first = False
        else:
            segment_gaps.append(now - last_yield)
        last_yield = now

        tts_module.play_text_async(sentence)

    wall_start_wait = time.time()
    tts_module.wait_until_done()
    tail_wait = time.time() - wall_start_wait

    total_time = time.time() - start_llm

    chunk_metrics = list(tts_module.chunk_metrics)
    silence_log = list(tts_module.silence_log)

    tts_module.flush_session_audio()

    return {
        "llm_first_token": first_token_time or 0.0,
        "llm_segment_gaps": segment_gaps,
        "tts_chunk_metrics": chunk_metrics,
        "silence_log": silence_log,
        "tail_wait": tail_wait,
        "total_time": total_time,
    }


def summarize(reps_data: list) -> dict:
    def mean_of(key_fn):
        vals = [key_fn(r) for r in reps_data]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else 0.0

    return {
        "llm_first_token_mean": mean_of(lambda r: r["llm_first_token"]),
        "llm_segment_gap_mean": mean_of(
            lambda r: statistics.mean(r["llm_segment_gaps"]) if r["llm_segment_gaps"] else None
        ),
        "tts_chunk_mean": mean_of(
            lambda r: statistics.mean([c["synth_time"] for c in r["tts_chunk_metrics"]])
            if r["tts_chunk_metrics"] else None
        ),
        "silence_total_mean": mean_of(lambda r: sum(r["silence_log"])),
        "total_time_mean": mean_of(lambda r: r["total_time"]),
    }


def print_table(rows: list):
    headers = ["Варіант", "LLM 1й (с)", "LLM наст. (с)", "TTS шматок (с)", "Тиша сум. (с)", "Всього (с)"]
    widths = [max(len(h), 14) for h in headers]
    widths[0] = max(widths[0], max(len(r[0]) for r in rows) if rows else 14)

    def fmt_row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        label, s = r
        print(fmt_row([
            label,
            f"{s['llm_first_token_mean']:.2f}",
            f"{s['llm_segment_gap_mean']:.2f}",
            f"{s['tts_chunk_mean']:.2f}",
            f"{s['silence_total_mean']:.2f}",
            f"{s['total_time_mean']:.2f}",
        ]))
    print()


def main():
    parser = argparse.ArgumentParser(description="Бенчмарк LLM+TTS пайплайна на фіксованій фразі")
    parser.add_argument("--phrase", default=DEFAULT_PHRASE, help="Фіксована фраза для всіх варіантів і повторів")
    parser.add_argument("--reps", type=int, default=3, help="Скільки повторів на варіант (дефолт 3)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Фіксований seed генерації LLM — та сама фраза + той самий seed = "
                              "той самий текст щоразу, інакше різниця між варіантами губиться в "
                              "шумі випадкового семплінгу. --seed 0 вимикає фіксацію (звичайна випадковість).")
    parser.add_argument("--mute", action="store_true", help="Не відтворювати звук по-справжньому (швидше, але менш точна 'тиша')")
    parser.add_argument("--csv", default=None, help="Куди зберегти CSV з усіма прогонами (за замовчуванням benchmark_<timestamp>.csv)")
    args = parser.parse_args()

    if args.mute:
        mute_playback()
        log.warning("🔇 [BENCH] Звук вимкнено (--mute) — 'тиша' вимірюється лише по завершенню синтезу, без реального відтворення.")

    if args.seed:
        log.info(f"🎲 [BENCH] Seed={args.seed} — LLM видаватиме той самий текст у кожному повторі/варіанті.")
    else:
        log.warning("🎲 [BENCH] Seed вимкнено (--seed 0) — LLM видаватиме різний текст щоразу, "
                    "різниця між варіантами буде змішана з випадковістю семплінгу.")

    if not os.path.exists(CONFIG_PATH):
        log.critical(f"❌ Конфігураційний файл {CONFIG_PATH} відсутній!")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    csv_path = args.csv or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_rows = []
    summary_rows = []

    for label, overrides in VARIANTS:
        log.info(f"\n{'=' * 60}\n🧪 [BENCH] Варіант: {label}\n{'=' * 60}")
        variant_config = deep_merge(base_config, overrides)

        llm_module = LLMHandler(variant_config)
        tts_module = TTSHandler(variant_config)

        reps_data = []
        try:
            for rep in range(1, args.reps + 1):
                log.info(f"  ▶ [BENCH] Повтор {rep}/{args.reps}...")
                llm_module.history = []  # чиста історія на кожен повтор, без переініціалізації моделі
                result = run_once(llm_module, tts_module, args.phrase, seed=(args.seed or None))
                reps_data.append(result)

                csv_rows.append({
                    "variant": label,
                    "rep": rep,
                    "llm_first_token": round(result["llm_first_token"], 3),
                    "llm_segment_gaps": ";".join(f"{g:.3f}" for g in result["llm_segment_gaps"]),
                    "tts_chunk_times": ";".join(f"{c['synth_time']:.3f}" for c in result["tts_chunk_metrics"]),
                    "silence_gaps": ";".join(f"{s:.3f}" for s in result["silence_log"]),
                    "total_time": round(result["total_time"], 3),
                })
        finally:
            tts_module.shutdown()
            # Явно звільняємо посилання на важкі моделі й форсуємо GC —
            # Python неохоче повертає пам'ять ОС сам по собі, а наступний
            # варіант має стартувати на максимально "чистій" пам'яті,
            # інакше залишки від важчого варіанта можуть зачепити своп і
            # спотворити заміри того, що йде ПІСЛЯ нього
            del llm_module, tts_module
            gc.collect()

        summary_rows.append((label, summarize(reps_data)))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    log.info(f"\n💾 [BENCH] Деталі кожного прогону збережено в {csv_path}")
    print_table(summary_rows)


if __name__ == "__main__":
    main()