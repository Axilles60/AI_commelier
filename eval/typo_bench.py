"""Опечатки в запросе: узнаёт ли разбор фильтров сорт/регион с одной ошибкой и не выдумывает ли фильтры на обычном
тексте. Без LLM: python -m eval.typo_bench

Recall: 40 самых частых сортов и все регионы каталога, по 3 опечатки (замена, пропуск или удвоение буквы в самом
длинном слове названия). Ложные срабатывания: реальные запросы из eval/teacher/topic/redteam и 600 предложений
статей портала — фильтр не должен появиться там, где без нечёткого поиска его не было.
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sommelier import articles, catalog, prefetch  # noqa: E402

ROOT = Path(__file__).resolve().parent
VOWELS = "аоеиуэыя"


def typo(word: str, rnd: random.Random) -> str:
    i = rnd.randrange(1, len(word) - 1)
    kind = rnd.choice(["sub", "drop", "dup"])
    if kind == "sub":
        low = word[i].lower()
        repl = rnd.choice([v for v in VOWELS if v != low]) if low in VOWELS else rnd.choice("бвгджзклмнпрст")
        return word[:i] + repl + word[i + 1:]
    return word[:i] + word[i + 1:] if kind == "drop" else word[:i] + word[i] + word[i:]


def recall(rnd: random.Random) -> dict:
    wines = catalog.load_catalog()["wines"]
    sets = {"grape": [g for g, _ in Counter(g for w in wines for g in w["grapes"]).most_common(40)],
            "region": [r for r, _ in Counter(w["region"] for w in wines if w.get("region")).most_common()]}
    out = {}
    for key, names in sets.items():
        ok = n = 0
        for name in names:
            for _ in range(3):
                words = name.split()
                j = max(range(len(words)), key=lambda x: len(words[x]))
                words[j] = typo(words[j], rnd)
                n += 1
                ok += prefetch.parse_filters("Хочу " + " ".join(words)).get(key) == name
        out[key] = (ok, n)
    return out


def false_positives(rnd: random.Random) -> dict:
    tasks = json.loads((ROOT / "tasks.json").read_text(encoding="utf-8"))
    queries = [t["query"] for t in tasks] + [f for t in tasks for f in t.get("followups", [])]
    queries += [q["query"] if isinstance(q, dict) else q
                for q in json.loads((ROOT / "teacher_queries.json").read_text(encoding="utf-8"))]
    src = (ROOT / "topic_bench.py").read_text(encoding="utf-8") + (ROOT / "redteam.py").read_text(encoding="utf-8")
    queries = list(dict.fromkeys(queries + re.findall(r'\("([^"]{8,})"', src)))
    sents = [s.strip() for c in articles.load_chunks() for s in re.split(r"[.!?\n]", c["text"]) if 30 < len(s.strip()) < 160]
    sample = rnd.sample(sents, min(600, len(sents)))

    def changed(texts):
        bad = []
        for q in texts:
            new = prefetch.parse_filters(q)
            with mock.patch.object(prefetch, "_fuzzy", lambda *a, **k: None):
                old = prefetch.parse_filters(q)
            if new != old:
                bad.append((q[:80], {k: v for k, v in new.items() if old.get(k) != v}))
        return bad
    return {"queries": (changed(queries), len(queries)), "articles": (changed(sample), len(sample))}


def main():
    rec = recall(random.Random(0))
    for k, (ok, n) in rec.items():
        print(f"опечатка в {k}: узнано {ok}/{n} ({ok / n:.0%})")
    with mock.patch.object(prefetch, "_fuzzy", lambda *a, **k: None):
        base = recall(random.Random(0))
    print("без нечёткого поиска: " + ", ".join(f"{k} {ok}/{n} ({ok / n:.0%})" for k, (ok, n) in base.items()))
    for name, (bad, n) in false_positives(random.Random(1)).items():
        print(f"ложные фильтры ({name}): {len(bad)}/{n}" + (f" — {bad[:5]}" if bad else ""))


if __name__ == "__main__":
    main()
