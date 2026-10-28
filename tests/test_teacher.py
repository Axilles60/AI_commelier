"""Разбор ответа teacher (помощник vino-svoe.ru) и reference-based метрика."""
from eval.teacher import compare, extract_wines
from sommelier import catalog

STEAK = [w["id"] for _, w, _ in catalog.search(dish="стейк", color="красное", sweetness="сухое", limit=3)]
SUSHI = [w["id"] for _, w, _ in catalog.search(dish="суши", color="белое", limit=3)]


def test_extracts_by_quoted_title_without_links():
    w0 = catalog.get_wine(STEAK[0])
    assert extract_wines(f"Попробуйте «{w0['title']}»")["ids"] == [w0["id"]]


def test_links_take_priority_and_broken_link_is_flagged():
    w1 = catalog.get_wine(STEAK[1])
    text = f"Вот «Red Blend»: {w1['url']} и https://vino-svoe.ru/wines/vinodelnya-marcheko-krasnostop-14"
    ex = extract_wines(text)
    assert ex["ids"] == [w1["id"]]
    assert ex["not_in_catalog"] == ["/wines/vinodelnya-marcheko-krasnostop-14"]


def test_unknown_wine_flagged_as_not_in_catalog():
    ex = extract_wines("Советую «Шато Несуществующее Гран Крю» к стейку, регион «Крым»")
    assert ex["ids"] == [] and ex["not_in_catalog"] == ["Шато Несуществующее Гран Крю"]


def test_grape_name_alone_is_not_a_wine():
    # «Пино Нуар» — и сорт, и название вина в каталоге; без кавычек это упоминание сорта
    assert extract_wines("Хорошо подойдёт пино нуар или каберне")["ids"] == []


def test_compare_rewards_same_style():
    same = compare(STEAK[:2], STEAK[1:3])["score"]
    other = compare(SUSHI, STEAK)["score"]
    assert same > other
    assert compare(STEAK, STEAK)["score"] == 1.0
    assert compare(STEAK, [])["score"] is None


def test_collect_refuses_mismatched_answer(tmp_path, monkeypatch):
    # ответ на «чужой» вопрос (виджет не успел очистить диалог) не должен попасть в датасет
    import pytest
    from eval import collect_teacher
    monkeypatch.setattr(collect_teacher, "DATASET", tmp_path / "ds.jsonl")
    with pytest.raises(SystemExit):
        collect_teacher.save("d01", {"q": "Хочу Рислинг", "text": "…", "urls": []})
    w = catalog.get_wine(STEAK[0])
    row = collect_teacher.save("d01", {"q": "Что выпить к шашлыку из баранины?", "text": "Вот", "urls": [w["url"]]})
    assert row["teacher_wine_ids"] == [w["id"]] and row["scenario"] == "dish"
    assert collect_teacher.load()["d01"]["teacher_wine_ids"] == [w["id"]]


def test_explicit_conditions_and_compliance():
    from eval.teacher_dataset_summary import compliance, explicit
    assert explicit("Мне нравится полусладкое красное") == {"color": "Красное", "sweetness": "полусладкое"}
    assert explicit("Вино к борщу и пельменям") == {"color": None, "sweetness": None}
    red_dry = [w["id"] for _, w, _ in catalog.search(color="красное", sweetness="сухое", limit=2)]
    rose = [w["id"] for _, w, _ in catalog.search(color="розовое", limit=2)]
    assert compliance(red_dry, explicit("Хочу красное")) == 1.0
    assert compliance(red_dry + rose, explicit("Хочу красное")) == 0.5  # как s01: просили красное, дали розовое
    assert compliance(red_dry, explicit("Вино к борщу")) is None
