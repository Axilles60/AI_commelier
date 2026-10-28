"""Детерминированный разбор фильтров из реплики — для предварительного поиска (prefetch) до вызова LLM.

Если фильтры однозначны («Красное сухое к шашлыку», «Хочу Красностоп», «Белое из Крыма»), поиск выполняется сразу
и кладётся в историю как обычный вызов search_wines: модель отвечает за один вызов вместо двух
(«решить искать» → «ответить»), а grounding_guard видит выдачу как раньше. На размытых запросах («Хочу вино»)
и на отрицаниях («не люблю сладкое») prefetch не срабатывает — там решает модель.
"""
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache

from . import articles, catalog
from .memory import DISLIKE, _mentions, _tokens

# только формы прилагательного: «Красностоп» или «Белый мускат» (сорта) цветом не считаются
ADJ = r"(ое|ая|ые|ый|ого|ому|ым|ую|ых|ой|ом|ими)"
COLOR_WORDS = [(rf"красн{ADJ}", "красное"), (rf"бел{ADJ}", "белое"), (rf"розов{ADJ}", "розовое"),
               (rf"оранжев{ADJ}", "оранжевое")]
STYLE_WORDS = {"хочу", "вино", "вина", "белое", "белого", "красное", "красного", "розовое", "сухое", "сухого",
               "полусухое", "сладкое", "полусладкое", "игристое", "брют", "оранжевое"}
NEG_HEAD = re.compile(r"\b(?:не|без|кроме)\s+(?:(?:хочу|хотим|надо|нужно|нужен|люблю|бери|берите|предлагай|"
                      r"советуй|стоит|буду|подойдет|подходит)\s+)?")


def _prefix(part: str) -> str:
    p = catalog.stem(part)
    return p if len(p) <= 5 else p[:max(5, len(p) - 2)]


def _grape_at(words: list):
    """Сорт, с которого начинаются слова: самое длинное название («Каберне Совиньон», а не «Каберне»)."""
    for name in sorted(catalog.filter_values("grapes"), key=lambda n: (-len(n.split()), -len(n))):
        parts = catalog.norm(name).split()
        if len(parts) <= len(words) and all(words[i].startswith(_prefix(p)) for i, p in enumerate(parts)):
            return name, len(parts)
    return None


def _strip_negations(clause: str, excluded: list) -> str:
    """Вырезает «не/без/кроме [хочу] X». Если X — сорт, вырезается всё название и сорт уходит в исключения:
    раньше «только не Каберне Совиньон» теряло лишь «Каберне», и оставшийся «Совиньон» становился фильтром —
    поиск искал ровно то, от чего отказались."""
    out, pos = [], 0
    for m in NEG_HEAD.finditer(clause):
        if m.start() < pos:
            continue
        words = list(re.finditer(r"[\wё-]+", clause[m.end():]))
        found = _grape_at([w.group(0) for w in words[:3]])
        if found:
            excluded.append(found[0])
        k = min(found[1] if found else 1, len(words))
        out.append(clause[pos:m.start()] + " ")
        pos = m.end() + (words[k - 1].end() if k else 0)
    return "".join(out) + clause[pos:]


def _positive_clauses(text: str, excluded: list | None = None) -> list:
    """Части реплики без отрицаний: «не люблю сладкое» выпадает целиком, а в «белое без дуба» вырезается
    только «без дуба» — «белое» остаётся фильтром. Сорта под отрицанием собираются в excluded."""
    excluded = [] if excluded is None else excluded
    clauses = re.split(r"[.!?;,]|\s+(?:и|а|но)\s+", catalog.norm(text))
    positive = []
    for c in clauses:
        if not c.strip():
            continue
        if DISLIKE.search(c):  # «не люблю Мерло» — вся часть не про то, что искать, но сорт исключаем
            words = re.findall(r"[\wё-]+", c)
            hits = [_grape_at(words[i:i + 3]) for i in range(len(words))]
            excluded += [h[0] for h in hits if h]
            continue
        positive.append(_strip_negations(c, excluded))
    return positive


def _first(pattern_list, clause: str):
    for pattern, value in pattern_list:
        if re.search(rf"(^|[^\wё]){pattern}(?![\wё])", clause):
            return value
    return None


FUZZY_CUTOFF = 0.82  # «Ризлинг», «Сапераве», «Каберне Соовиньон» — да; «Мерло» ~ «белое» (0.6) — нет


@lru_cache(maxsize=1)
def _known_words() -> frozenset:
    """Словарь настоящих слов (карточки вин и статьи портала, не реже двух раз): «сахара», «линейка»,
    «глубокого» — это слова, а не опечатки в «Самара», «индейка», «Голубок»."""
    texts = [w["_text"] for w in catalog.load_catalog()["wines"]] + [catalog.norm(c["text"]) for c in articles.load_chunks()]
    counts = Counter(t for text in texts for t in _tokens(text))
    return frozenset(t for t, c in counts.items() if c >= 2)


def _fuzzy(tokens: list, names, prefix_only: bool = False):
    """Запасной путь для опечаток: название сравнивается с окнами из стольких же слов реплики (и с их началом
    той же длины — падежные окончания не мешают). Берётся лучшее сходство не ниже FUZZY_CUTOFF. Окна только
    из настоящих слов пропускаются: опечатка — это слово, которого нет в словаре."""
    known = _known_words()
    best = (FUZZY_CUTOFF, 0, None)
    for name in names:
        target = catalog.norm(name)
        k = len(target.split())
        for i in range(len(tokens) - k + 1):
            window = " ".join(tokens[i:i + k])
            if len(window) < 5 or all(t in known for t in tokens[i:i + k]):
                continue
            if abs(len(window) - len(target)) > 4 and not prefix_only:
                continue
            cands = [window[:len(target)]] if prefix_only else [window, window[:len(target)]]
            if prefix_only and (cands[0][0], cands[0][-1]) != (target[0], target[-1]):
                continue  # основа блюда: опечатка внутри («шышлык»), а не другое слово («устроились» ≠ «устриц»)
            r = max(SequenceMatcher(None, c, target).ratio() for c in cands)
            if (r, len(target)) > best[:2]:
                best = (r, len(target), name)
    return best[2]


def parse_filters(text: str) -> dict:
    """{'color', 'sweetness', 'dish', 'region', 'grape', 'occasion'} — только то, что названо явно."""
    excluded = []
    clauses = _positive_clauses(text, excluded)
    joined = " ".join(clauses)
    tokens = _tokens(joined)
    f = {}
    color = next((c for c in (_first(COLOR_WORDS, cl) for cl in clauses) if c), None)
    sweet = next((s for s in (catalog.sweetness_of(cl) for cl in clauses) if s), None)
    if color:
        f["color"] = color
    if sweet:
        f["sweetness"] = sweet
    dish = (next((k for k in catalog.DISH_SYNONYMS if k in joined), None)
            or _fuzzy(tokens, [k for k in catalog.DISH_SYNONYMS if len(k) >= 5], prefix_only=True))
    if dish:
        f["dish"] = dish
    regions = [r for r in catalog.filter_values("region") if _mentions(tokens, r)]
    if len(regions) == 1:
        f["region"] = regions[0]
    grapes = sorted((g for g in catalog.filter_values("grapes") if _mentions(tokens, g)), key=len, reverse=True)
    if grapes:
        f["grape"] = grapes[0]
    rest = [t for t in tokens if t not in STYLE_WORDS]  # «белое», «сухое» не должны «похоже» совпасть с сортом
    region = None if regions else _fuzzy(rest, catalog.filter_values("region"))
    if region:
        f["region"] = region
    grape = None if grapes else _fuzzy(rest, catalog.filter_values("grapes"))
    if grape:
        f["grape"] = grape
    occasion = next((k for k in catalog.OCCASION_HINTS if k in joined), None)
    if occasion:
        f["occasion"] = occasion
    if excluded:
        f["exclude_grape"] = excluded[0]
    return f


CONTEXT_SLOTS = ("dish", "occasion")  # к чему и зачем вино; остальное — стиль (цвет, сладость, сорт, регион)


def merge_filters(prev: dict, cur: dict) -> dict:
    """Состояние диалога (dialog state tracking): ответ на уточнение («Белое, сухое» после «Подбери к рыбе»)
    дополняет прошлые условия, а не заменяет их; «Хочу белое» → «к рыбе» тоже. Другое блюдо или повод вместо уже
    названного — новый запрос: старое не переносится. Названное сейчас перекрывает прошлое; сорт и цвет
    взаимоисключают друг друга (Красностоп не бывает белым)."""
    new_context = any(k in cur for k in CONTEXT_SLOTS) and any(k in prev for k in CONTEXT_SLOTS)
    if not prev or not cur or new_context:
        return dict(cur)
    merged = dict(prev)
    if "grape" in cur:
        merged.pop("color", None)
    if "color" in cur:
        merged.pop("grape", None)
    merged.update(cur)
    return merged


def is_confident(f: dict) -> bool:
    """Prefetch, только если запрос конкретный: блюдо, сорт, регион или повод — или цвет вместе со сладостью."""
    return bool({"dish", "grape", "region", "occasion"} & f.keys()) or {"color", "sweetness"} <= f.keys()
