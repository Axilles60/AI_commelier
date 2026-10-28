"""Статьи портала «Своё Вино» как источник знаний для общих винных вопросов (термины, хранение, подача, культура).

Требование клиента: не выдавать информацию, кроме заявленной на сайте. Поэтому на вопрос «что такое терруар»
агент отвечает не из памяти модели, а по фрагментам статей портала — со ссылкой. Поиск — тот же IDF по основам
слов, что и в каталоге; слабые совпадения отбрасываются, чтобы не подгонять ответ под случайную статью.
"""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from . import catalog

ARTICLES_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.json"
CHUNK_CHARS = 700
MIN_SCORE = 3.0      # ниже — «на портале нет материала» (подобрано на примерах из tests/test_articles.py)
MIN_COVERAGE = 0.66  # фрагмент должен покрывать ≥ 2/3 значимых слов вопроса
# Клиент: только вино. Рубрики про другие напитки (глинтвейн, настойки на водке) в базу знаний не берём;
# новости и анонсы событий годятся хуже разборов — понижаем.
EXCLUDED_CATEGORIES = {"Коктейли", "Напитки", "Настойки"}
CATEGORY_WEIGHT = {"Новости": 0.6, "События": 0.6}
STOP = catalog.STOP | {"что", "такое", "как", "почему", "зачем", "какой", "какая", "какие", "ли", "это", "можно",
                       "нужно", "расскажи", "объясни", "чем", "где", "когда", "вином", "вина"}


@lru_cache(maxsize=1)
def load_chunks(path: str = str(ARTICLES_PATH)) -> list:
    p = Path(path)
    if not p.exists():
        return []
    chunks = []
    for a in json.loads(p.read_text(encoding="utf-8"))["articles"]:
        if a.get("category") in EXCLUDED_CATEGORIES:
            continue
        buf = ""
        for para in [x.strip() for x in a["text"].split("\n") if x.strip()]:
            if len(buf) + len(para) > CHUNK_CHARS and buf:
                chunks.append(_chunk(a, buf))
                buf = ""
            buf = f"{buf}\n{para}" if buf else para
        if buf:
            chunks.append(_chunk(a, buf))
    return chunks


def _chunk(a: dict, text: str) -> dict:
    return {"id": a["id"], "title": a["title"], "url": a["url"], "category": a.get("category"),
            "published": a.get("published"), "text": text, "_title": catalog.norm(a["title"]), "_text": catalog.norm(text)}


@lru_cache(maxsize=4096)
def _idf(term: str) -> float:
    chunks = load_chunks()
    n = len(chunks) or 1
    df = sum(term in c["_text"] or term in c["_title"] for c in chunks)
    return math.log((n - df + 0.5) / (df + 0.5) + 1)


@lru_cache(maxsize=4096)
def _in_corpus(term: str) -> bool:
    return any(term in c["_text"] or term in c["_title"] for c in load_chunks())


def query_terms(query: str) -> list:
    """Значимые слова вопроса. Латинские слова, которых нет нигде в статьях («decanting» — модель добавляет
    английский синоним), выбрасываются: совпасть они не могут, а покрытие роняли ниже порога, и «decanting
    декантация» не находило ничего. Незнакомое русское слово остаётся — это признак, что темы на портале нет."""
    words = re.findall(r"[\wё-]+", catalog.norm(query))
    terms = [catalog.stem(w) for w in words if w not in STOP and len(w) > 2]
    return [t for t in terms if not (re.fullmatch(r"[a-z-]+", t) and not _in_corpus(t))]


def search(query: str, k: int = 3) -> list:
    """Лучшие фрагменты (не больше одного на статью): [{title, url, category, published, snippet, score}]."""
    terms = query_terms(query)
    if not terms:
        return []
    scored = []
    for c in load_chunks():
        hits = [t for t in terms if t in c["_text"] or t in c["_title"]]
        coverage = len(hits) / len(terms)
        if not hits or coverage < MIN_COVERAGE:
            continue
        score = sum(_idf(t) * (2.0 if t in c["_title"] else 1.0) for t in hits)
        score *= coverage * CATEGORY_WEIGHT.get(c["category"], 1.0)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for score, c in scored:
        if score < MIN_SCORE:
            break
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append({"id": c["id"], "title": c["title"], "url": c["url"], "category": c["category"],
                    "published": c["published"], "snippet": c["text"][:CHUNK_CHARS], "score": round(score, 2)})
        if len(out) == k:
            break
    return out
