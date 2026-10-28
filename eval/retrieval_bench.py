"""Бенчмарк поиска «известного элемента» (known-item search) — офлайн, без LLM.

Для 200 случайных вин запрос = 4 случайных содержательных слова из описания; смотрим, на каком месте
это вино в catalog.search(query=...). Метрики: Recall@1, Recall@10, MRR@10.
Сравнение режимов взвешивания: python -m eval.retrieval_bench
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sommelier import catalog  # noqa: E402

STOPWORDS = {"вкус", "аромат", "ароматы", "вино", "оттенки", "нотки", "ноты", "тонами", "послевкусие", "цвет",
             "который", "которое", "очень", "также", "этого", "этот", "более"}


def make_queries(n=200, k=4, seed=7):
    rnd = random.Random(seed)
    wines = [w for w in catalog.load_catalog()["wines"] if len(w.get("description") or "") > 120]
    out = []
    for w in rnd.sample(wines, n):
        words = [x for x in re.findall(r"[а-яё]{5,}", catalog.norm(w["description"])) if x not in STOPWORDS]
        if len(set(words)) >= k:
            out.append((w["id"], " ".join(rnd.sample(sorted(set(words)), k))))
    return out


def make_noisy_queries(n=200, seed=11):
    """Реалистичнее: 2 характерных слова из карточки + 2 частых слова, которых в карточке НЕТ
    (пользователь описывает вкус и своими словами, и общими штампами)."""
    rnd = random.Random(seed)
    cat = catalog.load_catalog()["wines"]
    common = ["свежий", "легкий", "яркий", "насыщенный", "фруктовый", "мягкий", "элегантный", "гармоничный",
              "ягодный", "пряный", "цветочный", "сбалансированный"]
    out = []
    for w in rnd.sample([x for x in cat if len(x.get("description") or "") > 120], n):
        words = sorted({x for x in re.findall(r"[а-яё]{5,}", catalog.norm(w["description"])) if x not in STOPWORDS})
        rare = sorted(words, key=catalog._idf, reverse=True)[:6]
        absent = [c for c in common if catalog.stem(c) not in w["_text"]]
        if len(rare) >= 2 and len(absent) >= 2:
            out.append((w["id"], " ".join(rnd.sample(rare, 2) + rnd.sample(absent, 2))))
    return out


def evaluate(queries):
    r1 = r10 = mrr = 0.0
    for wid, q in queries:
        ids = [w["id"] for _, w, _ in catalog.search(query=q, limit=10)]
        if wid in ids:
            rank = ids.index(wid) + 1
            r10 += 1
            r1 += rank == 1
            mrr += 1 / rank
    n = len(queries)
    return {"recall@1": round(r1 / n, 3), "recall@10": round(r10 / n, 3), "mrr@10": round(mrr / n, 3), "n": n}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for name, qs in (("4 слова из карточки", make_queries()), ("2 характерных + 2 чужих", make_noisy_queries())):
        for mode in ("count", "idf"):
            catalog.QUERY_WEIGHTING = mode
            catalog.term_weight.cache_clear()
            print(f"{name:24} {mode:5}", evaluate(qs))
