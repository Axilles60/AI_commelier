"""Детерминированное извлечение предпочтений и нормализация профиля."""
import pytest

from sommelier import memory
from sommelier.memory import extract_preferences


@pytest.mark.parametrize("text,expected", [
    ("Я обожаю вина Долины Дона и не люблю сладкое", [("liked_regions", "Долина Дона"), ("avoid", "сладкое")]),
    ("Люблю сухое красное из Крыма", [("liked_regions", "Крым"), ("liked_categories", "Красное сухое"),
                                       ("liked_sweetness", "сухое")]),
    ("Предпочитаю брют", [("liked_sweetness", "брют")]),
    ("Не люблю полусладкое, люблю сухие", [("avoid", "полусладкое"), ("liked_sweetness", "сухое")]),
    ("Мне нравится Пино Нуар", [("liked_grapes", "Пино Нуар")]),
    ("Обожаю кубанские и ставропольские вина", [("liked_regions", "Кубань"), ("liked_regions", "Ставрополье")]),
    ("Не люблю Мерло", [("avoid", "Мерло")]),
    ("терпеть не могу дуб", [("avoid", "дуб")]),
])
def test_extracts(text, expected):
    assert extract_preferences(text) == expected


@pytest.mark.parametrize("text", ["Хочу вино к стейку", "Нужен подарок", "Люблю красное",
                                  "Что-нибудь к ужину, как я люблю", "Самая лучшая бутылка"])
def test_no_false_positives(text):
    # «красное» не должно стать сортом «Красностоп», «самая» — регионом «Самара»
    assert extract_preferences(text) == []


def test_canonical_casing_matches_catalog():
    memory.add_to_list("c", "liked_categories", "красное сухое")
    memory.add_to_list("c", "liked_regions", "крым")
    prof = memory.load_profile("c")
    assert prof["liked_categories"] == ["Красное сухое"] and prof["liked_regions"] == ["Крым"]


def test_parallel_tool_calls_do_not_lose_profile_data():
    # регрессия из eval t11: ToolNode пишет профиль из нескольких потоков; раньше гонка обнуляла профиль
    from concurrent.futures import ThreadPoolExecutor
    values = [f"сорт{i}" for i in range(40)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda v: memory.add_to_list("race", "liked_grapes", v), values))
    assert sorted(memory.load_profile("race")["liked_grapes"]) == sorted(values)


def test_sweetness_canonical():
    memory.add_to_list("s", "liked_sweetness", "сухие")
    memory.add_to_list("s", "liked_sweetness", "Сухое")
    assert memory.load_profile("s")["liked_sweetness"] == ["сухое"]


@pytest.mark.parametrize("text", ["Белое, сухое", "Красное сухое, к шашлыку", "Игристое на свадьбу", "Давай розовое"])
def test_situational_choice_is_not_a_taste(text):
    """Ответ на уточнение — выбор на этот раз, а не вкус: в долгосрочный профиль он не попадает."""
    assert memory.extract_preferences(text) == []


def test_disliked_grape_does_not_spill_to_its_substring_grape():
    """«Не люблю Каберне Совиньон» — это не «не люблю Совиньон» (а значит, и не Совиньон Блан)."""
    assert extract_preferences("Не люблю Каберне Совиньон") == [("avoid", "Каберне Совиньон")]
    liked = [v for k, v in extract_preferences("Обожаю Красностоп Золотовский") if k == "liked_grapes"]
    assert liked == ["Красностоп Золотовский"]
