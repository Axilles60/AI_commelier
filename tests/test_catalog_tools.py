"""Точность поиска и отсутствие выдуманных позиций на уровне инструментов (детерминированно)."""
import json

import pytest

from sommelier import catalog, memory
from sommelier.tools import get_wine_details, record_feedback, search_wines, update_taste_profile

CFG = {"configurable": {"user_id": "tester"}}


def run_search(**kw):
    return json.loads(search_wines.invoke(kw, config=CFG))["results"]


def test_catalog_is_real_and_complete():
    data = catalog.load_catalog()
    assert data["meta"]["source"] == "https://vino-svoe.ru"
    assert len(data["wines"]) >= 2000
    ids = [w["id"] for w in data["wines"]]
    assert len(ids) == len(set(ids))
    for w in data["wines"]:
        assert w["url"] == f"https://vino-svoe.ru/wines/{w['id']}"
        assert w["title"] and w["category"]
        assert "price" not in w  # цен на сайте нет — и в каталоге их быть не должно


@pytest.mark.parametrize("kw", [
    dict(dish="стейк"), dict(dish="суши", color="белое"), dict(color="красное", sweetness="сухое", region="Крым"),
    dict(grape="Красностоп"), dict(region="Долина Дона", color="красное"), dict(occasion="новый год"),
    dict(query="ягодный выдержка в дубе", color="красное"), dict(dish="сырная тарелка", color="розовое"),
    dict(sweetness="сладкое", dish="десерт"), dict(manufacturer="Фанагория"),
])
def test_every_result_exists_in_catalog_and_matches_filters(kw):
    res = run_search(**kw, limit=10)
    assert res, f"пустая выдача для {kw}"
    for r in res:
        w = catalog.get_wine(r["id"])
        assert w is not None, "выдуманное вино"
        assert r["title"] == w["title"] and r["url"] == w["url"]  # поля не искажены
        if kw.get("color"):
            assert w["category"].lower().startswith(kw["color"])
        if kw.get("sweetness"):
            assert w["category"].lower().endswith(kw["sweetness"])
        if kw.get("region"):
            assert w["region"] == kw["region"]
        if kw.get("grape"):
            assert any("красностоп" in g.lower() for g in w["grapes"])
        if kw.get("manufacturer"):
            assert kw["manufacturer"].lower() in w["manufacturer"].lower()
        if kw.get("dish"):
            targets = catalog.dish_targets(kw["dish"])
            assert any(t in catalog.norm(d) for d in w["dishes"] for t in targets), "блюдо не из карточки"


def test_dish_pairing_comes_from_site_data():
    for r in run_search(dish="стейк", limit=10):
        assert "Мясо и стейки" in catalog.get_wine(r["id"])["dishes"]
        assert any("Мясо и стейки" in x for x in r["match_reasons"])


def test_occasion_prefers_sparkling():
    res = run_search(occasion="праздник", limit=5)
    assert all("брют" in r["category"].lower() for r in res)


def test_no_results_is_honest():
    out = json.loads(search_wines.invoke({"region": "Бордо", "color": "красное"}, config=CFG))
    assert out["results"] == [] and "не придумывайте" in out["note"]


def test_unknown_wine_id_rejected_by_schema():
    with pytest.raises(Exception, match="нет в каталоге"):
        get_wine_details.invoke({"wine_id": "chateau-margaux-2015"})
    with pytest.raises(Exception, match="нет в каталоге"):
        record_feedback.invoke({"wine_id": "fake-wine", "verdict": "liked"}, config=CFG)


def test_details_are_verbatim_catalog_fields():
    wid = run_search(grape="Красностоп")[0]["id"]
    d = json.loads(get_wine_details.invoke({"wine_id": wid}))
    w = catalog.get_wine(wid)
    for k in ("title", "description", "grapes", "dishes", "region", "manufacturer"):
        assert d.get(k) == w.get(k) or (not w.get(k) and k not in d)


def test_dislike_excludes_wine_and_like_personalizes():
    base = run_search(color="красное", sweetness="сухое", limit=3)
    first = base[0]["id"]
    record_feedback.invoke({"wine_id": first, "verdict": "disliked"}, config=CFG)
    assert first not in [w["id"] for _, w, _ in catalog.search(color="красное", sweetness="сухое", profile=memory.load_profile("tester"), limit=5000)]

    # понравился красностоп → он поднимается в выдаче «красное сухое»
    kr = run_search(grape="Красностоп", color="красное", sweetness="сухое")[0]["id"]
    record_feedback.invoke({"wine_id": kr, "verdict": "liked"}, config=CFG)
    prof = memory.load_profile("tester")
    assert kr in prof["liked"] and prof["liked_grapes"]
    top = run_search(color="красное", sweetness="сухое", limit=5)
    liked_grapes = set(prof["liked_grapes"])
    assert all(liked_grapes & set(catalog.get_wine(r["id"])["grapes"]) for r in top)
    assert any("профиле" in " ".join(r["match_reasons"]) for r in top)


def test_taste_profile_avoid_lowers_rank():
    update_taste_profile.invoke({"key": "avoid", "value": "дуб"}, config=CFG)
    top = run_search(color="красное", sweetness="сухое", limit=5)
    assert all("дуб" not in catalog.get_wine(r["id"])["_text"] for r in top)


def test_profiles_are_isolated_per_user():
    update_taste_profile.invoke({"key": "liked_regions", "value": "Крым"}, config=CFG)
    assert memory.load_profile("someone_else") == {}


# --- сладость в профиле -------------------------------------------------------
PINOT_CRIMEA = {"liked_grapes": ["Пино Нуар"], "liked_regions": ["Крым"]}


def _sweet_of_top(profile, n=6, **kw):
    return [w["_sweet"] for _, w, _ in catalog.search(profile=profile, limit=n, **kw)]


def test_unknown_sweetness_prefers_typical_style_not_alphabet():
    # регрессия t12: раньше при равных баллах наверх попадало полусладкое «AGORA» по алфавиту id
    assert set(_sweet_of_top(PINOT_CRIMEA, grape="Пино Нуар", region="Крым")) == {"сухое"}


def test_typical_style_is_data_driven_not_always_dry():
    # к десерту типичный стиль — сладкий, приор не должен навязывать сухое
    assert "сухое" not in _sweet_of_top({}, n=5, dish="десерт", color="белое")


def test_explicit_sweetness_preference_wins():
    prof = dict(PINOT_CRIMEA, liked_sweetness=["полусладкое"])
    top = catalog.search(grape="Пино Нуар", region="Крым", profile=prof, limit=4)
    assert {w["_sweet"] for _, w, _ in top} == {"полусладкое"}
    assert all("как вы любите" in " ".join(r) for _, _, r in top)


def test_liked_dry_category_implies_dry_preference():
    # неявная обратная связь: понравилось «Красное сухое» → сухое выше и в розовых
    top = _sweet_of_top({"liked_categories": ["Красное сухое"]}, n=5, color="розовое")
    assert set(top) == {"сухое"}


def test_avoid_sweet_excludes_semisweet_too():
    top = _sweet_of_top({"avoid": ["сладкое"]}, n=15, color="красное", dish="десерт")
    assert not {"сладкое", "полусладкое"} & set(top[:5])


def test_avoid_sweet_does_not_penalize_dry_wines_with_sweet_words():
    # «ноты сладких специй» в описании сухого вина — не повод его штрафовать
    plain = [w["id"] for _, w, _ in catalog.search(color="красное", sweetness="сухое", limit=20)]
    avoided = [w["id"] for _, w, _ in catalog.search(color="красное", sweetness="сухое", profile={"avoid": ["сладкое"]},
                                                     limit=20)]
    assert plain == avoided


def test_explicit_sweetness_filter_is_not_overridden_by_prior():
    assert set(_sweet_of_top(PINOT_CRIMEA, sweetness="полусладкое", grape="Пино Нуар")) == {"полусладкое"}


# --- качество поиска (eval/retrieval_bench.py) ---------------------------------
def test_rare_word_outweighs_common():
    from sommelier.catalog import term_weight
    # «кизил» — в 42 карточках, «дуб» — в 178, «аромат» — в 1479
    assert term_weight("кизил") > term_weight("дуб") > 2 * term_weight("аромат")


def test_retrieval_quality_on_noisy_queries():
    # без IDF: Recall@10 = 0.045 — общие слова («свежий», «ягодный») топили нужное вино
    from eval.retrieval_bench import evaluate, make_noisy_queries
    assert evaluate(make_noisy_queries(n=100))["recall@10"] >= 0.6


@pytest.mark.parametrize("dish", ["форель", "судак", "сельдь", "икра", "гусь", "кролик", "плов", "ризотто", "оливье",
                                  "шаурма", "фондю", "паэлья", "тартар", "том ям", "лазанья", "хачапури", "долма",
                                  "оленина", "блины", "тирамису", "чизкейк", "орехи", "брынза", "фуа-гра"])
def test_popular_dishes_are_recognized(dish):
    # раньше 29 из 38 популярных блюд давали пустую выдачу
    res = catalog.search(dish=dish, limit=3)
    assert res and all("рекомендовано к" in " ".join(r) for _, _, r in res)


def test_semi_dry_is_not_tagged_as_dry():
    """Регрессия: «полусухое» оканчивается на «сухое», и 105 полусухих вин считались сухими —
    запрос «полусухое» не находил ничего, а «сухое» подмешивал полусухие."""
    from collections import Counter
    wines = catalog.load_catalog()["wines"]
    assert all(w["_sweet"] == "полусухое" for w in wines if w["category"].lower().endswith("полусухое"))
    assert catalog.search(sweetness="полусухое", limit=5)
    assert all("полусух" not in w["category"].lower() for _, w, _ in catalog.search(sweetness="сухое", limit=50))
    assert Counter(w["_sweet"] for w in wines)["полусухое"] >= 100


def test_search_tool_honours_exclude_grape():
    import json
    from sommelier.tools import search_wines
    out = json.loads(search_wines.invoke({"color": "красное", "sweetness": "сухое", "exclude_grape": "Мерло", "limit": 10},
                                         config={"configurable": {"user_id": "ex1"}}))
    assert out["results"] and not any("мерло" in g.lower() for r in out["results"] for g in r["grapes"])
