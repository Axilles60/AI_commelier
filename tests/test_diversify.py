"""MMR-разнообразие выдачи."""
from sommelier import catalog
from sommelier.diversify import mmr, similarity


def w(manufacturer, grapes, category="Красное сухое"):
    return {"manufacturer": manufacturer, "grapes": grapes, "category": category}


def test_similarity_prefers_same_winery():
    a = w("Усадьба Перовских", ["Мерло"])
    assert similarity(a, w("Усадьба Перовских", ["Сира"])) > similarity(a, w("Фанагория", ["Мерло"])) > 0


def test_mmr_breaks_ties_towards_other_wineries():
    results = [(5.0, w("A", ["Мерло"]), []), (5.0, w("A", ["Мерло"]), []), (5.0, w("A", ["Сира"]), []),
               (5.0, w("B", ["Каберне"]), []), (5.0, w("C", ["Саперави"]), [])]
    picked = mmr(results, 3)
    assert [r[1]["manufacturer"] for r in picked] == ["A", "B", "C"]


def test_mmr_keeps_clearly_better_wine_first():
    results = [(9.0, w("A", ["Мерло"]), []), (8.9, w("A", ["Мерло"]), []), (2.0, w("B", ["Сира"]), [])]
    assert mmr(results, 2)[0][0] == 9.0 and mmr(results, 2)[1][0] == 8.9  # огромный разрыв в релевантности не жертвуем


def test_search_top5_is_diverse_and_still_filtered():
    res = catalog.search(dish="стейк", color="красное", sweetness="сухое", limit=5)
    assert len({x["manufacturer"] for _, x, _ in res}) >= 4
    assert all(x["category"] == "Красное сухое" and "Мясо и стейки" in x["dishes"] for _, x, _ in res)
