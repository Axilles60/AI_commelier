"""Инструменты агента (Лекция 1.2 Tools): единственный канал фактов о винах — каталог.

Pydantic-схемы валидируют аргументы до вызова (как BookingRequest в Семинаре 2):
несуществующий wine_id отклоняется ещё до выполнения инструмента.
"""
import json
from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from . import articles, catalog, memory


def _user(config: RunnableConfig) -> str:
    return (config or {}).get("configurable", {}).get("user_id", "anon")


def _check_wine_id(v: str) -> str:
    if v not in catalog.wine_ids():
        raise ValueError(f"Вина '{v}' нет в каталоге «Своё Вино». Используйте id из результатов search_wines.")
    return v


class SearchArgs(BaseModel):
    query: str = Field("", description="Свободный текст: вкус, ароматы, стиль ('лёгкое фруктовое', 'выдержка в дубе')")
    color: Literal["", "красное", "белое", "розовое", "оранжевое"] = ""
    sweetness: Literal["", "экстра брют", "брют", "сухое", "полусухое", "полусладкое", "сладкое"] = ""
    region: str = Field("", description="Регион России: Крым, Кубань, Долина Дона, Дагестан, Нижняя Волга, ...")
    grape: str = Field("", description="Сорт винограда, напр. 'Красностоп', 'Рислинг'")
    manufacturer: str = Field("", description="Винодельня")
    dish: str = Field("", description="Блюдо, к которому нужно вино: 'стейк', 'суши', 'сырная тарелка'")
    occasion: str = Field("", description="Повод: подарок, праздник, свидание, пикник, новый год")
    exclude_grape: str = Field("", description="Сорт, который пользователь просил исключить: «только не Мерло», "
                                               "«кроме Каберне Совиньон», «не люблю Саперави»")
    limit: int = Field(5, ge=1, le=10)


@tool(args_schema=SearchArgs)
def search_wines(query: str = "", color: str = "", sweetness: str = "", region: str = "", grape: str = "",
                 manufacturer: str = "", dish: str = "", occasion: str = "", limit: int = 5,
                 exclude_grape: str = "", config: RunnableConfig = None) -> str:
    """Искать вина в реальном каталоге «Своё Вино». Учитывает вкусовой профиль пользователя.
    Возвращает JSON со списком вин (id, название, категория, сорта, регион, блюда с сайта) и причинами выбора."""
    profile = memory.load_profile(_user(config))
    hits = catalog.search(query, color, sweetness, region, grape, manufacturer, dish, occasion, profile, limit,
                          exclude_grape=exclude_grape)
    if not hits:
        return json.dumps({"results": [], "note": "По этим условиям в каталоге ничего не найдено. "
                           "Предложите ослабить фильтр — не придумывайте вино."}, ensure_ascii=False)
    return json.dumps({"results": [dict(catalog.public_view(w), match_reasons=r) for _, w, r in hits],
                       "note": "Цен и наличия в каталоге нет."}, ensure_ascii=False)


class WineIdArgs(BaseModel):
    wine_id: str = Field(description="id вина из результатов search_wines")

    @field_validator("wine_id")
    @classmethod
    def validate_wine_id(cls, v: str) -> str:
        return _check_wine_id(v)


@tool(args_schema=WineIdArgs)
def get_wine_details(wine_id: str) -> str:
    """Полная карточка вина из каталога: описание, крепость, температура подачи, блюда, ссылка."""
    return json.dumps(catalog.public_view(catalog.get_wine(wine_id), short=False), ensure_ascii=False)


@tool
def list_catalog_values(field: Literal["region", "grapes", "manufacturer", "category", "dishes"]) -> str:
    """Список допустимых значений поля в каталоге (регионы, сорта, винодельни, категории, блюда)."""
    vals = catalog.filter_values(field)
    return json.dumps({"field": field, "count": len(vals), "values": vals[:150]}, ensure_ascii=False)


class ProfileArgs(BaseModel):
    key: Literal["liked_grapes", "liked_regions", "liked_categories", "liked_sweetness", "avoid", "name", "experience"]
    value: str


@tool(args_schema=ProfileArgs)
def update_taste_profile(key: str, value: str, config: RunnableConfig = None) -> str:
    """Сохранить предпочтение пользователя в долгосрочный вкусовой профиль
    (любимые сорта/регионы/категории, любимая сладость — liked_sweetness: сухое/брют/полусладкое…,
    что не нравится — avoid, имя, опыт)."""
    uid = _user(config)
    if key in memory.LIST_FIELDS:
        memory.add_to_list(uid, key, value)
    else:
        memory.update_profile(uid, lambda prof: prof.__setitem__(key, value))
    return f"Профиль обновлён: {key} += '{value}'"


class FeedbackArgs(BaseModel):
    wine_id: str
    verdict: Literal["liked", "disliked", "saved"]
    comment: Optional[str] = None

    @field_validator("wine_id")
    @classmethod
    def validate_wine_id(cls, v: str) -> str:
        return _check_wine_id(v)


@tool(args_schema=FeedbackArgs)
def record_feedback(wine_id: str, verdict: str, comment: Optional[str] = None, config: RunnableConfig = None) -> str:
    """Сохранить обратную связь по вину: liked / disliked / saved (в избранное).
    Понравившееся вино обновляет любимые сорта и регион; не понравившееся больше не предлагается."""
    uid = _user(config)
    memory.add_to_list(uid, verdict, wine_id)
    w = catalog.get_wine(wine_id)
    if verdict == "liked":
        for g in w["grapes"]:
            memory.add_to_list(uid, "liked_grapes", g)
        if w.get("region"):
            memory.add_to_list(uid, "liked_regions", w["region"])
        memory.add_to_list(uid, "liked_categories", w["category"])
    return f"Записано: {verdict} — {w['title']}"


@tool
def get_saved_wines(config: RunnableConfig = None) -> str:
    """Показать сохранённые и понравившиеся пользователю вина из каталога."""
    prof = memory.load_profile(_user(config))
    out = {k: [catalog.public_view(catalog.get_wine(i)) for i in prof.get(k, []) if catalog.get_wine(i)]
           for k in ("saved", "liked")}
    return json.dumps(out, ensure_ascii=False)


@tool
def search_articles(query: str) -> str:
    """Найти фрагменты статей портала «Своё Вино» для общего вопроса о вине: термины, хранение, подача, технологии,
    культура, российское виноделие. Отвечай ТОЛЬКО по найденным фрагментам и давай ссылку на статью."""
    found = articles.search(query, k=3)
    if not found:
        return json.dumps({"results": [], "note": "На портале нет статьи по этому вопросу. Скажи об этом честно и "
                                                  "предложи подобрать вино — не отвечай из общих знаний."},
                          ensure_ascii=False)
    return json.dumps({"results": [{k: a[k] for k in ("title", "url", "published", "snippet")} for a in found],
                       "note": "Перескажи коротко своими словами только то, что есть во фрагментах, со ссылкой url."},
                      ensure_ascii=False)


TOOLS = [search_wines, get_wine_details, list_catalog_values, update_taste_profile, record_feedback, get_saved_wines,
         search_articles]
