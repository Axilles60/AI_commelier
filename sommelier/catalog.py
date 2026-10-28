"""Каталог «Своё Вино» + детерминированный поиск (retrieval-слой, как lookup_policy в Семинаре 2).

Единственный источник фактов о вине — data/catalog.json, выгруженный с vino-svoe.ru.
Поиск keyword-based с простым стеммингом и словарём синонимов блюд/событий.
"""
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

COLORS = {"красное": "Красное", "белое": "Белое", "розовое": "Розовое", "оранжевое": "Оранжевое"}
SWEETNESS = ["экстра брют", "брют", "сухое", "полусухое", "полусладкое", "сладкое"]

# Разговорные блюда → категории блюд, которые использует сайт (поле dishes).
# Ключи — основы слов (stem), значения — подстроки названий блюд на сайте.
DISH_SYNONYMS = {
    "стейк": ["мясо и стейки"], "говяд": ["мясо и стейки"], "мяс": ["мясо и стейки", "мясное ассорти"],
    "шашлык": ["bbq", "мясо и стейки", "кавказская"], "барбекю": ["bbq"], "bbq": ["bbq"], "гриль": ["bbq", "овощи гриль"],
    "баран": ["мясо и стейки", "кавказская"], "ягнят": ["мясо и стейки"], "свинин": ["мясо и стейки", "bbq"],
    "колбас": ["мясное ассорти"], "хамон": ["мясное ассорти"], "паштет": ["паштеты"],
    "утк": ["блюда из птицы"], "куриц": ["блюда из птицы"], "индейк": ["блюда из птицы"], "птиц": ["блюда из птицы"],
    "рыб": ["блюда из рыбы", "рыба и морепродукты"], "лосос": ["блюда из рыбы"], "семг": ["блюда из рыбы"],
    "суши": ["азиатская кухня", "рыба и морепродукты"], "ролл": ["азиатская кухня", "рыба и морепродукты"],
    "азиат": ["азиатская кухня"], "кревет": ["морепродукты", "рыба и морепродукты"], "мид": ["морепродукты"],
    "устриц": ["устрицы", "морепродукты"], "морепрод": ["морепродукты", "рыба и морепродукты"],
    "сыр": ["сыры"], "паст": ["паста"], "пицц": ["пицца"], "салат": ["салаты"], "брускет": ["брускетты"],
    "овощ": ["свежие овощи", "запеченные овощи", "овощи гриль"], "остр": ["острое"], "хинкал": ["кавказская кухня"],
    "кавказ": ["кавказская кухня"], "пельмен": ["русская кухня"], "борщ": ["русская кухня"], "русск": ["русская кухня"],
    "бургер": ["фастфуд"], "фастфуд": ["фастфуд"], "чипс": ["фастфуд", "легкие закуски"],
    "десерт": ["десерты", "выпечка и десерты"], "торт": ["выпечка и десерты", "десерты"], "мороже": ["мороженое"],
    "шоколад": ["шоколад"], "фрукт": ["фрукты", "фруктово-ягодные десерты"], "ягод": ["фруктово-ягодные десерты"],
    "выпечк": ["выпечка и десерты", "несладкая выпечка"], "пирог": ["несладкая выпечка", "выпечка и десерты"],
    "закуск": ["закуски", "легкие закуски"], "аперитив": ["легкие закуски"], "средиземн": ["средиземноморская кухня"],
    # популярные блюда, которые раньше не находились вовсе (проверка: 29 из 38 давали пустую выдачу)
    "форел": ["блюда из рыбы", "рыба и морепродукты"], "судак": ["блюда из рыбы", "рыба и морепродукты"],
    "треск": ["блюда из рыбы"], "сельд": ["блюда из рыбы", "русская кухня"], "селедк": ["блюда из рыбы", "русская кухня"],
    "тунец": ["блюда из рыбы", "азиатская кухня"], "дорадо": ["блюда из рыбы"], "сибас": ["блюда из рыбы"],
    "икр": ["морепродукты", "закуски"], "гус": ["блюда из птицы"], "кролик": ["блюда из птицы", "мясо и стейки"],
    "оленин": ["мясо и стейки"], "кабан": ["мясо и стейки"], "плов": ["кухни народов мира", "мясо и стейки"],
    "ризотто": ["средиземноморская кухня", "паста"], "лазань": ["паста"], "спагетти": ["паста"], "макарон": ["паста"], "болоньез": ["паста"],
    "карбонар": ["паста"], "феттучин": ["паста"], "пенне": ["паста"], "равиол": ["паста"],
    "томатн": ["паста", "пицца"],  # «под томатный соус» (симулятор: блюдо не узнавалось, подбор уходил от пасты)
    "паэль": ["средиземноморская кухня", "морепродукты"],
    "оливье": ["салаты", "русская кухня"], "блин": ["русская кухня", "выпечка и десерты"], "шаурм": ["фастфуд"],
    "фондю": ["сыры"], "брынз": ["сыры"], "хачапури": ["кавказская кухня", "несладкая выпечка", "сыры"],
    "долм": ["кавказская кухня"], "тартар": ["мясо и стейки", "закуски"], "карпаччо": ["мясо и стейки", "закуски"],
    "том ям": ["азиатская кухня", "острое"], "рамен": ["азиатская кухня"], "карри": ["азиатская кухня", "острое"],
    "тирамису": ["десерты", "выпечка и десерты"], "чизкейк": ["десерты", "выпечка и десерты"],
    "орех": ["легкие закуски", "сыры"], "фуа": ["паштеты"],
}

# События/поводы → фильтры по категории (это правила подбора, а не свойства вин).
OCCASION_HINTS = {
    "праздн": {"prefer_category": ["брют", "экстра брют", "игрист"]},
    "новый год": {"prefer_category": ["брют", "экстра брют", "игрист"]},
    "свадьб": {"prefer_category": ["брют", "игрист"]},
    "день рожден": {"prefer_category": ["брют", "игрист"]},
    "свидан": {"prefer_category": ["розовое", "брют", "игрист"]},
    "пикник": {"prefer_category": ["розовое", "белое"]},
    "лето": {"prefer_category": ["розовое", "белое"]},
}


def norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def stem(word: str) -> str:
    """Грубый стемминг для русского: отрезаем окончание, оставляем основу >=4 символов."""
    w = norm(word)
    w = re.sub(r"(ами|ями|ого|его|ому|ему|ыми|ими|ая|яя|ое|ее|ые|ие|ой|ей|ий|ый|ам|ям|ах|ях|ов|ев|ом|ем|ы|и|а|я|е|у|ю|о)$", "", w)
    return w if len(w) >= 3 else norm(word)


STOP = {"вино", "вина", "вин", "мне", "для", "под", "к", "и", "или", "что", "на", "с", "в", "по",
        "хочу", "нужно", "нужен", "посоветуй", "подбери", "какое", "какой", "бутылк", "бутылка", "что-нибудь"}


@lru_cache(maxsize=1)
def load_catalog(path: str = str(CATALOG_PATH)) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for w in data["wines"]:
        cat = norm(w.get("category"))
        w["_color"] = next((v for k, v in COLORS.items() if cat.startswith(k)), None)
        # длинные названия раньше коротких: «полусухое» тоже оканчивается на «сухое»
        w["_sweet"] = next((s for s in sorted(SWEETNESS, key=len, reverse=True) if cat.endswith(s)), None)
        w["_text"] = norm(" ".join([w.get("title") or "", w.get("category") or "", w.get("manufacturer") or "",
                                     w.get("region") or "", " ".join(w["grapes"]), " ".join(w["dishes"]),
                                     w.get("description") or ""]))
    data["by_id"] = {w["id"]: w for w in data["wines"]}
    return data


def wine_ids() -> set:
    return set(load_catalog()["by_id"])


def get_wine(wine_id: str):
    return load_catalog()["by_id"].get(wine_id)


def public_view(w: dict, short: bool = True) -> dict:
    """Только поля из каталога — никаких вычисленных «свойств», которые LLM могла бы выдать за факт."""
    keys = ["id", "title", "category", "manufacturer", "region", "grapes", "dishes", "temperature", "url"]
    if not short:
        keys += ["color_name", "alcohol", "alcohol_max", "description", "rating"]
    out = {k: w.get(k) for k in keys if w.get(k) not in (None, "", [])}
    if short and w.get("description"):
        out["description_excerpt"] = w["description"][:220]
    return out


def dish_targets(dish: str) -> list:
    d = norm(dish)
    targets = set()
    for key, vals in DISH_SYNONYMS.items():
        if key in d:
            targets.update(vals)
    if not targets and d:
        targets.add(stem(d))
    return sorted(targets)


TYPICAL_STYLE_BONUS = 0.1  # только разбивает ничьи: не должен перевешивать совпадение по тексту запроса
DIVERSIFY = True  # MMR-реранкинг топа (eval/diversity_bench.py)
QUERY_WEIGHTING = "idf"  # "count" — каждое совпавшее слово +1 (как было); "idf" — редкие слова весят больше


@lru_cache(maxsize=4096)
def term_weight(term: str) -> float:
    """Вес слова запроса по редкости в каталоге (BM25-подобный IDF), в диапазоне 0.3–2.0.
    «аромат» есть почти в каждой карточке и почти не различает вина, «кизил» — сильный сигнал."""
    if QUERY_WEIGHTING == "count":
        return 1.0
    return _idf(term)


@lru_cache(maxsize=4096)
def _idf(term: str) -> float:
    wines = load_catalog()["wines"]
    n = len(wines)
    df = sum(term in w["_text"] for w in wines)
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    return round(0.3 + 1.7 * idf / math.log(n + 1), 3)


def _match(value: str, field_value: str) -> bool:
    return stem(value) in norm(field_value) if value else True


# порядок важен: «полусладкое» проверяем раньше «сладкого», «экстра брют» раньше «брюта»
SWEETNESS_STEMS = [("экстра брют", "экстра брют"), ("брют", "брют"), ("полусух", "полусухое"),
                   ("полусладк", "полусладкое"), ("сух", "сухое"), ("сладк", "сладкое")]


def sweetness_of(text: str):
    """«сухие» → «сухое», «Красное полусладкое» → «полусладкое»; None, если сладость не названа."""
    t = norm(text)
    return next((v for k, v in SWEETNESS_STEMS if re.search(rf"(^|[^\wё]){k}", t)), None)


def sweetness_prefs(profile: dict):
    """Профиль → (любимая сладость, нелюбимая сладость, прочие «не люблю» для поиска по тексту).

    Любимая — явная liked_sweetness плюс сладость понравившихся категорий (неявная обратная связь).
    «Не люблю сладкое» относится к категории вина, а не к слову в описании: иначе штрафовались бы
    сухие вина с «нотами сладких специй»; и «сладкое» включает «полусладкое».
    """
    liked = {s for s in (sweetness_of(v) for v in profile.get("liked_sweetness", [])
                         + profile.get("liked_categories", [])) if s}
    avoid, text_avoid = set(), []
    for a in profile.get("avoid", []):
        s = sweetness_of(a)
        if s:
            avoid |= {"сладкое", "полусладкое"} if s == "сладкое" else {s}
        else:
            text_avoid.append(a)
    return liked - avoid, avoid, text_avoid


def search(query: str = "", color: str = "", sweetness: str = "", region: str = "", grape: str = "",
           manufacturer: str = "", dish: str = "", occasion: str = "", profile: dict | None = None,
           limit: int = 5, exclude_grape: str = "") -> list:
    """Фильтрация + ранжирование. Возвращает список (score, wine, reasons) — reasons строятся из полей каталога."""
    cat = load_catalog()
    profile = profile or {}
    disliked = set(profile.get("disliked", []))
    liked_sweet, avoid_sweet, text_avoid = sweetness_prefs(profile)
    q_words = [stem(w) for w in re.findall(r"[\wё-]+", norm(query)) if norm(w) not in STOP and len(w) > 2]
    dish_t = dish_targets(dish) if dish else []
    occ = next((v for k, v in OCCASION_HINTS.items() if k in norm(occasion)), {}) if occasion else {}

    results = []
    for w in cat["wines"]:
        if w["id"] in disliked:
            continue
        # жёсткие фильтры
        if color and w["_color"] != COLORS.get(norm(color), color.capitalize()):
            continue
        if sweetness and w["_sweet"] != norm(sweetness):
            continue
        if region and not _match(region, w.get("region")):
            continue
        if grape and not any(_match(grape, g) for g in w["grapes"]):
            continue
        if exclude_grape and any(_match(exclude_grape, g) for g in w["grapes"]):  # «только не Каберне Совиньон»
            continue
        if manufacturer and not _match(manufacturer, w.get("manufacturer")):
            continue
        score, reasons = 0.0, []
        if dish_t:
            hits = [d for d in w["dishes"] if any(t in norm(d) for t in dish_t)]
            if not hits:
                continue
            score += 3 + len(hits)
            reasons.append(f"на сайте рекомендовано к: {', '.join(hits)}")
        if grape:
            reasons.append(f"сорт: {', '.join(g for g in w['grapes'] if _match(grape, g))}")
        if region:
            reasons.append(f"регион: {w['region']}")
        q_hits = [qw for qw in q_words if qw in w["_text"]]
        score += sum(term_weight(qw) for qw in q_hits)
        if q_hits:
            reasons.append(f"в карточке есть: {', '.join(q_hits)}")
        if q_words and not dish_t and score == 0 and not any([color, sweetness, region, grape, manufacturer]):
            continue
        if occ:
            if any(p in norm(w["category"]) for p in occ["prefer_category"]):
                score += 2
                reasons.append(f"категория «{w['category']}» подходит к поводу")
        # персонализация (долгосрочная память): бонус за сорта/регионы/категории из профиля
        for g in w["grapes"]:
            if g in profile.get("liked_grapes", []):
                score += 1.5
                reasons.append(f"сорт {g} есть в вашем профиле")
        if w.get("region") in profile.get("liked_regions", []):
            score += 1
            reasons.append(f"регион {w['region']} вам нравится")
        if w.get("category") in profile.get("liked_categories", []):
            score += 1
        if w["_sweet"] in avoid_sweet:
            score -= 5
        elif w["_sweet"] in liked_sweet:
            score += 1.5
            reasons.append(f"сладость «{w['_sweet']}» — как вы любите")
        if any(a and stem(a) in w["_text"] for a in text_avoid):
            score -= 3
        if w["id"] in profile.get("saved", []):
            score += 0.5
        if w.get("description"):
            score += 0.2  # предпочитаем карточки с описанием — объяснение будет обоснованным
        results.append((round(score, 2), w, reasons))

    if results and not liked_sweet and not sweetness:
        # сладость пользователя неизвестна: равные по баллу вина раньше разводил алфавит id,
        # и в «Пино Нуар из Крыма» наверх попадало полусладкое (5 из 100), хотя типичный стиль — сухое (40 из 100).
        # Теперь мягко предпочитаем самый частый стиль среди подходящих вин.
        typical = Counter(w["_sweet"] for _, w, _ in results).most_common(1)[0][0]
        results = [(round(s + TYPICAL_STYLE_BONUS, 2) if w["_sweet"] == typical else s, w, r) for s, w, r in results]

    results.sort(key=lambda x: (-x[0], x[1]["id"]))
    if DIVERSIFY and limit > 1:
        from .diversify import mmr  # разные винодельни/сорта вместо трёх вин одного производителя
        return mmr(results, limit)
    return results[:limit]


def filter_values(field: str) -> list:
    cat = load_catalog()["wines"]
    if field == "grapes":
        vals = {g for w in cat for g in w["grapes"]}
    elif field == "dishes":
        vals = {d for w in cat for d in w["dishes"]}
    else:
        vals = {w.get(field) for w in cat if w.get(field)}
    return sorted(vals)
