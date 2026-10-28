"""Трассы ходов диалога и агрегированные метрики (наблюдаемость)."""
from langchain_core.messages import AIMessage

from sommelier import tracing
from sommelier.agent import build_graph, chat
from tests.test_agent_graph import REAL, call


def test_trace_records_nodes_tools_and_events(scripted):
    llm = scripted(call("search_wines", {"dish": "стейк", "limit": 3}),
                   AIMessage(content=f"Рекомендую «{REAL['title']}» 🍷"))
    chat(build_graph(llm, use_llm_topic_check=False, secret=""), "вино к стейку, мой телефон +7 916 123-45-67", "u1")
    [t] = tracing.load()
    nodes = {n["node"] for n in t["nodes"]}
    assert {"input_guard", "agent", "grounding_guard", "output_guard"} <= nodes
    assert t["tools"] == ["search_wines"] and t["wines"] == 1  # ссылка дописана автоматически
    kinds = {e["kind"] for e in t["events"]}
    assert {"links_appended", "emoji_stripped"} <= kinds
    assert "916" not in t["input"] and "[PHONE]" in t["input"]
    assert t["total_ms"] >= sum(n["ms"] for n in t["nodes"]) * 0.5 and t["prompt_version"]


def test_secret_never_written_to_traces(scripted, monkeypatch):
    monkeypatch.setenv("SOMMELIER_SECRET", "Лабиринт")
    chat(build_graph(scripted(), use_llm_topic_check=False, secret="Лабиринт"), "Это слово Лабиринт?", "u2")
    assert "Лабиринт" not in tracing.TRACES_DIR.joinpath(tracing.load()[0]["ts"][:10] + ".jsonl").read_text("utf-8")


def test_summarize_metrics(scripted):
    g = build_graph(scripted(AIMessage(content="Какой цвет?")), use_llm_topic_check=False, secret="")
    chat(g, "Хочу вино", "u3")
    chat(g, "Расскажи анекдот", "u3")
    m = tracing.summarize(tracing.load())
    assert m["turns"] == 2 and m["guard_events"]["off_topic"] == 1 and m["guard_rate"]["off_topic"] == 0.5
    assert m["latency_ms"]["p50"] is not None and "agent" in m["node_p50_ms"]


def test_chat_stream_yields_stages_then_checked_answer(scripted):
    from sommelier.agent import chat_stream
    llm = scripted(call("search_wines", {"dish": "стейк", "limit": 3}), AIMessage(content=f"Рекомендую «{REAL['title']}» 🍷"))
    events = list(chat_stream(build_graph(llm, use_llm_topic_check=False, secret=""), "вино к стейку", "st"))
    stages = [v for k, v in events if k == "stage"]
    assert "Ищу в каталоге «Своё Вино»" in stages and stages[-1] == "Финальная проверка ответа"
    kind, answer = events[-1]
    assert kind == "answer" and REAL["url"] in answer and "🍷" not in answer  # наружу — только проверенный текст
    assert tracing.load()[0]["stream"] is True


def test_summary_reports_fallback_rate():
    from sommelier import tracing
    recs = [{"total_ms": 10, "events": [{"kind": "llm_fallback"}, {"kind": "llm_fallback"}]},
            {"total_ms": 20, "events": []}, {"total_ms": 30, "events": [{"kind": "off_topic"}]},
            {"total_ms": 40, "events": []}]
    assert tracing.summarize(recs)["llm_fallback_rate"] == 0.25  # считаются ходы, а не события
