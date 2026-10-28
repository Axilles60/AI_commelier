"""Статьи портала как единственный источник знаний для общих винных вопросов."""
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from sommelier import articles
from sommelier.agent import build_graph, chat, check_grounding

pytestmark = pytest.mark.skipif(not articles.load_chunks(), reason="нет data/articles.json (scripts/fetch_articles.py)")


@pytest.mark.parametrize("question,title_word", [
    ("Чем брют отличается от экстра брюта?", "брют"),
    ("Что значит полусухое вино?", "полусухое"),
    ("Как правильно открыть игристое?", "открыть"),
    ("Что такое декантация?", "декантац"),
    ("Какие автохтонные сорта есть в России?", "автохтон"),
])
def test_knowledge_questions_find_dedicated_articles(question, title_word):
    top = articles.search(question, k=1)[0]
    assert title_word in top["title"].lower() and top["url"].startswith("https://vino-svoe.ru/articles/")


def test_other_drinks_are_not_in_knowledge_base():
    assert articles.search("Какую настойку сделать на водке?") == []
    assert not any(c["category"] in articles.EXCLUDED_CATEGORIES for c in articles.load_chunks())


def test_article_answer_is_grounded(scripted):
    from tests.test_agent_graph import call
    top = articles.search("Что такое декантация?", k=1)[0]
    llm = scripted(call("search_articles", {"query": "декантация вина"}),
                   AIMessage(content=f"Декантация — переливание вина в графин. Подробнее: {top['url']}"))
    out = chat(build_graph(llm, use_llm_topic_check=False, secret=""), "Что такое декантация вина?", "a1")
    assert top["url"] in out and "Проверка качества" not in llm.seen[-1][-1].content


def test_invented_article_link_is_caught():
    tm = ToolMessage(content=json.dumps({"results": []}), tool_call_id="x")
    issues = check_grounding("См. https://vino-svoe.ru/articles/vydumannaya-statya", [tm])
    assert issues and "статью" in issues[0]


def test_english_synonym_from_model_does_not_kill_search():
    # модель иногда добавляет в запрос английский синоним: «decanting» нет в статьях и раньше обнулял выдачу
    assert articles.search("decanting декантация", k=1)
    assert articles.search("Какую настойку сделать на водке?") == []  # незнакомое русское слово — по-прежнему «нет темы»


def test_quoted_term_from_article_is_not_a_fake_wine():
    """«Брют, экстра-брют, брют натюр» из статьи — термин, а не выдуманное вино (раньше — ревизия и даже fallback)."""
    top = articles.search("брют и экстра брют разница сахар игристое вино", k=2)
    tm = ToolMessage(content=json.dumps({"results": top}, ensure_ascii=False), tool_call_id="x", name="search_articles")
    assert check_grounding("«Брют, экстра-брют, брют натюр» различаются сахаром. " + top[0]["url"], [tm]) == []


def test_invented_wine_from_real_card_words_is_still_caught():
    from sommelier import catalog
    w = catalog.search(grape="Красностоп", limit=1)[0][1]
    tm = ToolMessage(content=json.dumps({"results": [catalog.public_view(w)]}, ensure_ascii=False),
                     tool_call_id="y", name="search_wines")
    assert check_grounding("Рекомендую «Красностоп Резерв Особый» https://vino-svoe.ru/wines/" + w["id"], [tm])


def test_fallback_for_article_question_gives_article_links(scripted):
    from tests.test_agent_graph import call
    top = articles.search("Что такое декантация?", k=1)[0]
    bad = "См. https://vino-svoe.ru/articles/vydumannaya-statya"  # дважды мимо выдачи
    llm = scripted(call("search_articles", {"query": "декантация"}), AIMessage(content=bad), AIMessage(content=bad))
    out = chat(build_graph(llm, use_llm_topic_check=False, secret=""), "Что такое декантация вина?", "fa1")
    assert top["url"] in out and "цвет и сладость" not in out
