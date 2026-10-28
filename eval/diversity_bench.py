"""Разнообразие топ-5 до и после MMR на типовых запросах (без LLM).

Метрики: среднее число разных виноделен в топ-5, доля выдач, где ≥2 вина одной винодельни,
и «цена» разнообразия — средний балл релевантности топ-5 относительно исходного топ-5.
  python -m eval.diversity_bench
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sommelier import catalog  # noqa: E402
from sommelier.diversify import mmr  # noqa: E402

QUERIES = [dict(dish="стейк"), dict(dish="стейк", color="красное", sweetness="сухое"), dict(dish="суши", color="белое"),
           dict(grape="Красностоп"), dict(region="Крым", color="красное", sweetness="сухое"), dict(occasion="новый год"),
           dict(dish="десерт"), dict(color="розовое", dish="закуски"), dict(grape="Пино Нуар"), dict(region="Долина Дона"),
           dict(dish="утка"), dict(dish="сыр"), dict(grape="Саперави"), dict(region="Дагестан"), dict(dish="шашлык"),
           dict(query="ягодный выдержка в дубе", color="красное"), dict(dish="рыба", color="белое", sweetness="сухое"),
           dict(region="Кубань", color="белое"), dict(dish="паста"), dict(grape="Рислинг")]


def stats(lists, base_lists):
    distinct = [len({w["manufacturer"] for _, w, _ in lst}) for lst in lists]
    dup = [len(lst) - d > 0 for lst, d in zip(lists, distinct)]
    rel = [sum(s for s, _, _ in lst) / max(1e-9, sum(s for s, _, _ in base)) for lst, base in zip(lists, base_lists)]
    return {"разных виноделен в топ-5": round(statistics.mean(distinct), 2), "выдач с повтором винодельни": sum(dup),
            "релевантность к исходной": round(statistics.mean(rel), 3)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    base, diverse = [], []
    # MMR встроен в catalog.search: для исходной выдачи его нужно выключить, иначе сравнивается MMR с MMR
    catalog.DIVERSIFY = False
    for q in QUERIES:
        full = catalog.search(**q, limit=30)
        base.append(full[:5])
        diverse.append(mmr(full, 5))
    print("до MMR   ", stats(base, base))
    print("после MMR", stats(diverse, base))


if __name__ == "__main__":
    main()
