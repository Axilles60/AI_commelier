"""Запасная модель: сбой основной у провайдера не превращается в ошибку для клиента."""
import pytest
from langchain_core.messages import AIMessage

from sommelier import tracing
from sommelier.agent import build_graph, chat, fallback_model_name
from tests.conftest import ScriptedLLM


class DownLLM(ScriptedLLM):
    """Основная модель «упала»: каждый вызов — ошибка провайдера."""
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(messages)
        raise ConnectionError("provider unavailable")


def test_agent_and_topic_check_fall_back_to_backup_model(scripted):
    primary = DownLLM(script=[], seen=[])
    backup = scripted(AIMessage(content="yes"), AIMessage(content="К какому блюду подобрать вино?"))
    out = chat(build_graph(primary, use_llm_topic_check=True, secret="", fallback_llm=backup), "Хочу вино", "fb")
    assert out == "К какому блюду подобрать вино?"
    assert len(primary.seen) == 2 and len(backup.seen) == 2  # классификатор темы и агент — оба через запасную
    events = [e for e in tracing.load()[0]["events"] if e["kind"] == "llm_fallback"]
    assert [e["node"] for e in events] == ["topic", "agent"] and events[0]["error"] == "ConnectionError"


def test_without_backup_error_reaches_server_handler(scripted):
    with pytest.raises(ConnectionError):
        chat(build_graph(DownLLM(script=[], seen=[]), use_llm_topic_check=False, secret=""), "Хочу красное", "fb2")


def test_fallback_model_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("MODEL_NAME", "qwen/qwen3.8-flash")
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    assert fallback_model_name() == "qwen/qwen3.7-flash"
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "")
    assert fallback_model_name() == ""                       # явно выключено
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "qwen/qwen3.8-flash")
    assert fallback_model_name() == ""                       # запасная = основная — бессмысленно


def test_circuit_breaker_opens_then_probes():
    from sommelier.agent import CircuitBreaker
    now = [0.0]
    b = CircuitBreaker(threshold=2, cooldown=60, clock=lambda: now[0])
    b.record(False)
    assert b.allow()                      # одна ошибка — ещё не повод
    b.record(False)
    assert not b.allow()                  # две подряд — пауза
    now[0] = 61
    assert b.allow()                      # пауза прошла — пробная попытка
    b.record(False)
    assert not b.allow()                  # проба не удалась — сразу снова пауза
    now[0] = 130
    b.record(True)
    assert b.allow() and b.failures == 0  # удалась — работаем как обычно


def test_open_circuit_skips_primary_on_next_turns(scripted):
    primary = DownLLM(script=[], seen=[])
    backup = scripted(AIMessage(content="yes"), AIMessage(content="Какой цвет?"),
                      AIMessage(content="yes"), AIMessage(content="Какое блюдо?"))
    g = build_graph(primary, use_llm_topic_check=True, secret="", fallback_llm=backup)
    chat(g, "Хочу вино", "cb")
    assert len(primary.seen) == 2         # первый ход: классификатор и агент упали — предохранитель сработал
    chat(g, "Хочу вино к ужину", "cb")
    assert len(primary.seen) == 2         # второй ход: основную модель даже не ждём (раньше ~40 с повторов)
    kinds = [e.get("error") for e in tracing.load()[-1]["events"] if e["kind"] == "llm_fallback"]
    assert kinds == ["circuit_open", "circuit_open"]


def test_breaker_state_is_exposed_for_metrics(scripted):
    primary = DownLLM(script=[], seen=[])
    g = build_graph(primary, use_llm_topic_check=True, secret="",
                    fallback_llm=scripted(AIMessage(content="yes"), AIMessage(content="Какой цвет?")))
    assert {k: b.state for k, b in g.breakers.items()} == {"agent": "closed", "topic": "closed"}
    chat(g, "Хочу вино", "cbm")
    assert g.breakers["agent"].state == "open"
