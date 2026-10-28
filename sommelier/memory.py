"""Долгосрочная память: вкусовой профиль пользователя (аналог passenger_profile из Семинара 2).

Краткосрочная память — MemorySaver-чекпоинтер по thread_id (см. agent.py).
Запись в профиль идёт двумя путями: LLM вызывает update_taste_profile, а явные фразы
(«обожаю вина Дона», «не люблю сладкое») дополнительно ловит детерминированный extract_preferences.
"""
import json
import os
import re
import threading
from pathlib import Path

from . import catalog

PROFILES_DIR = Path(os.getenv("SOMMELIER_PROFILES_DIR", Path(__file__).resolve().parent.parent / "data" / "profiles"))

LIST_FIELDS = {"liked_grapes", "liked_regions", "liked_categories", "liked_sweetness", "avoid", "saved", "liked",
               "disliked"}
SCALAR_FIELDS = {"name", "budget_note", "experience"}

CANONICAL_FIELDS = {"liked_grapes": "grapes", "liked_regions": "region", "liked_categories": "category"}


def _path(user_id: str) -> Path:
    safe = "".join(c for c in user_id if c.isalnum() or c in "-_") or "anon"
    return PROFILES_DIR / f"{safe}.json"


def load_profile(user_id: str) -> dict:
    p = _path(user_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}  # не затираем файл: битым он мог «показаться» во время чужой записи
    return {}


def save_profile(user_id: str, profile: dict) -> None:
    """Атомарная запись: читатель никогда не увидит наполовину записанный JSON."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(user_id)
    tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


# ToolNode выполняет параллельные tool calls в потоках: read-modify-write профиля должен быть под замком
_LOCK = threading.RLock()


def update_profile(user_id: str, fn) -> dict:
    with _LOCK:
        prof = load_profile(user_id)
        fn(prof)
        save_profile(user_id, prof)
        return prof


def canonical(key: str, value: str) -> str:
    """«красное сухое» → «Красное сухое», «крым» → «Крым»: ранжирование сравнивает с полями каталога точно."""
    if key == "liked_sweetness":
        return catalog.sweetness_of(value) or value
    field = CANONICAL_FIELDS.get(key)
    if not field:
        return value
    nv = catalog.norm(value).strip()
    return next((v for v in catalog.filter_values(field) if catalog.norm(v) == nv), value)


def add_to_list(user_id: str, key: str, value: str) -> dict:
    value = canonical(key, value)

    def change(prof):
        lst = prof.setdefault(key, [])
        if value not in lst:
            lst.append(value)
        # like и dislike взаимоисключающие
        opposite = {"liked": "disliked", "disliked": "liked"}.get(key)
        if opposite and value in prof.get(opposite, []):
            prof[opposite].remove(value)

    return update_profile(user_id, change)


def remove_from_list(user_id: str, key: str, value: str) -> dict:
    """Удаление со страниц «Профиль» и «Избранное»."""
    def change(prof):
        if value in prof.get(key, []):
            prof[key] = [v for v in prof[key] if v != value]
        if not prof.get(key):
            prof.pop(key, None)
    return update_profile(user_id, change)


def favorites(user_id: str) -> list:
    """Избранное без дублей: вино, которое и «в избранном», и «понравилось», показывается один раз с обеими метками."""
    prof = load_profile(user_id)
    saved, liked = prof.get("saved", []), prof.get("liked", [])
    out = []
    for wid in dict.fromkeys(saved + liked):
        w = catalog.get_wine(wid)
        if w:
            out.append(dict(catalog.public_view(w), saved=wid in saved, liked=wid in liked))
    return out


def delete_profile(user_id: str) -> bool:
    """Право на удаление (152-ФЗ): стираем профиль, избранное и скрытые вина пользователя."""
    with _LOCK:
        p = _path(user_id)
        existed = p.exists()
        p.unlink(missing_ok=True)
        return existed


def hidden(user_id: str) -> list:
    """Вина, отмеченные «Не нравится»: поиск их больше не предлагает, на странице профиля их можно вернуть."""
    return [catalog.public_view(catalog.get_wine(w)) for w in load_profile(user_id).get("disliked", [])
            if catalog.get_wine(w)]


def profile_as_text(profile: dict) -> str:
    if not profile:
        return "  (пусто — пока ничего не сохранено)"
    return "\n".join(f"  {k}: {v}" for k, v in profile.items())


# --- детерминированное извлечение предпочтений ---------------------------------

LIKE = re.compile(r"(?<!\bне )\b(люблю|обожаю|нравится|нравятся|предпочитаю|фанат)")
DISLIKE = re.compile(r"\bне\s+(люблю|нравится|нравятся|пью|переношу)|терпеть не могу")
REGION_ALIASES = {"осет": "Северная Осетия — Алания", "алани": "Северная Осетия — Алания",
                  "дальн": "Дальневосточная зона", "примор": "Дальневосточная зона"}
TASTE_WORDS = [("полусладк", "полусладкое"), ("сладк", "сладкое"), ("кисл", "кислое"),
               ("дуб", "дуб"), ("танин", "танины"), ("терпк", "терпкое")]
COLOR_WORDS = [("красн", "Красное"), ("бел", "Белое"), ("розов", "Розовое"), ("оранжев", "Оранжевое")]
SWEET_WORDS = [("экстра брют", "экстра брют"), ("брют", "брют"), ("полусух", "полусухое"),
               ("полусладк", "полусладкое"), ("сух", "сухое"), ("сладк", "сладкое")]


def _tokens(clause: str) -> list:
    return re.findall(r"[\wё]+", clause)


def _mentions(tokens: list, name: str) -> bool:
    """Все значимые слова названия встречаются в реплике (с учётом падежей: сравниваем по префиксу основы)."""
    parts = [catalog.stem(p) for p in _tokens(catalog.norm(name)) if len(p) >= 3]
    # длинная основа может отличаться только окончанием: «красностоп» не должен совпасть с «красное»
    prefixes = [p if len(p) <= 5 else p[:max(5, len(p) - 2)] for p in parts]
    return bool(parts) and all(any(t.startswith(p) for t in tokens) for p in prefixes)


def _most_specific(names: list) -> list:
    """«Каберне Совиньон» совпадает и с сортом «Совиньон»: оставляем только самое полное название. Иначе «не люблю
    Каберне Совиньон» записывало бы в «не люблю» и «Совиньон» — и штраф получал бы Совиньон Блан, другой сорт."""
    words = {n: set(catalog.norm(n).split()) for n in names}
    return [n for n in names if not any(words[n] < words[m] for m in names)]


def _category(clause: str):
    color = next((v for k, v in COLOR_WORDS if re.search(rf"\b{k}", clause)), None)
    sweet = next((v for k, v in SWEET_WORDS if re.search(rf"\b{k}", clause)), None)
    return f"{color} {sweet}" if color and sweet else None


def extract_preferences(text: str) -> list:
    """«Я обожаю вина Долины Дона и не люблю сладкое» → [(liked_regions, Долина Дона), (avoid, сладкое)]."""
    out = []
    clauses = []
    for sentence in re.split(r"[.!?;]", catalog.norm(text)):
        mode = None  # «обожаю кубанские и ставропольские» — глагол относится к обеим частям
        for clause in re.split(r",|\s+(?:и|а|но)\s+", sentence):
            mode = "dislike" if DISLIKE.search(clause) else "like" if LIKE.search(clause) else mode
            clauses.append((clause, mode))
    for clause, mode in clauses:
        tokens = _tokens(clause)
        grapes = _most_specific([g for g in catalog.filter_values("grapes") if _mentions(tokens, g)])
        if mode == "dislike":
            taste = next((v for k, v in TASTE_WORDS if k in clause), None)
            out += [("avoid", taste)] if taste else []
            out += [("avoid", g) for g in grapes]
        elif mode == "like":
            regions = {v for k, v in REGION_ALIASES.items() if any(t.startswith(k) for t in tokens)}
            regions |= {r for r in catalog.filter_values("region") if _mentions(tokens, r)}
            out += [("liked_regions", r) for r in sorted(regions)]
            out += [("liked_grapes", g) for g in grapes]
            cat = _category(clause)
            out += [("liked_categories", cat)] if cat else []
            sweet = catalog.sweetness_of(clause)
            out += [("liked_sweetness", sweet)] if sweet else []
    return list(dict.fromkeys(out))
