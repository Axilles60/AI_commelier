"""Наблюдаемость (Лекция 5 «Production Engineering»): трасса каждого хода диалога.

Что пишется в data/traces/ГГГГ-ММ-ДД.jsonl:
  время каждого узла графа, вызванные инструменты, события guardrails (off-topic, critic, утечки, эмодзи),
  общая задержка, версия промпта и модели. Вход пользователя — с маскированием PII и секретного слова.
Агрегаты (p50/p95 задержки, частота срабатываний защит) отдаёт summarize() → /api/metrics.
"""
import contextvars
import datetime
import functools
import hashlib
import inspect
import json
import os
import re
import statistics
import threading
import time
import uuid
from pathlib import Path

TRACES_DIR = Path(os.getenv("SOMMELIER_TRACES_DIR", Path(__file__).resolve().parent.parent / "data" / "traces"))
_CURRENT = contextvars.ContextVar("sommelier_trace", default=None)
_LOCK = threading.Lock()


def event(kind: str, **data) -> None:
    """Отметить событие в текущей трассе (если трасса открыта)."""
    trace = _CURRENT.get()
    if trace is not None:
        trace["events"].append({"kind": kind, **data})


def timed(name: str, fn):
    """Обёртка узла графа: замеряет время. Сохраняет сигнатуру — LangGraph по ней решает, передавать ли config."""
    takes_config = len(inspect.signature(fn).parameters) > 1

    def record(start):
        trace = _CURRENT.get()
        if trace is not None:
            trace["nodes"].append({"node": name, "ms": round((time.perf_counter() - start) * 1000, 1)})

    if takes_config:
        @functools.wraps(fn)
        def wrapper(state, config):
            start = time.perf_counter()
            try:
                return fn(state, config)
            finally:
                record(start)
    else:
        @functools.wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            try:
                return fn(state)
            finally:
                record(start)
    return wrapper


class Trace:
    """with Trace(...) as t: ... — открывает трассу хода, по выходу дописывает её в JSONL."""

    def __init__(self, user_id: str, thread_id: str, text: str, mask, meta: dict):
        self.data = {"trace_id": uuid.uuid4().hex[:12], "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                     "user": hashlib.sha256(user_id.encode()).hexdigest()[:10], "thread": thread_id,
                     "input": mask(text)[:300], "nodes": [], "events": [], **meta}
        self._token = None

    def __enter__(self):
        self._start = time.perf_counter()
        self._token = _CURRENT.set(self.data)
        return self

    def finish(self, messages, answer: str):
        from langchain_core.messages import AIMessage
        self.data["tools"] = [c["name"] for m in messages if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
        self.data["answer_chars"] = len(answer)
        self.data["wines"] = len(set(re.findall(r"vino-svoe\.ru/wines/([\w-]+)", answer)))

    def __exit__(self, exc_type, exc, tb):
        self.data["total_ms"] = round((time.perf_counter() - self._start) * 1000, 1)
        if exc is not None:
            self.data["error"] = f"{exc_type.__name__}: {exc}"[:300]
        _CURRENT.reset(self._token)
        write(self.data)
        return False


def write(record: dict) -> None:
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACES_DIR / f"{record['ts'][:10]}.jsonl"
    with _LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load(day: str | None = None) -> list:
    path = TRACES_DIR / f"{day or datetime.date.today().isoformat()}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(records: list) -> dict:
    """Агрегаты для дашборда/алертов: задержка, срабатывания защит, инструменты, медленные узлы."""
    if not records:
        return {"turns": 0}
    total = [r["total_ms"] for r in records if "total_ms" in r]
    kinds = {}
    for r in records:
        for e in r.get("events", []):
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    per_node = {}
    for r in records:
        for n in r.get("nodes", []):
            per_node.setdefault(n["node"], []).append(n["ms"])
    return {
        "turns": len(records),
        "errors": sum("error" in r for r in records),
        "latency_ms": {"p50": _pct(total, 0.5), "p95": _pct(total, 0.95), "max": max(total) if total else None},
        "guard_events": kinds,
        "guard_rate": {k: round(v / len(records), 3) for k, v in kinds.items()},
        "avg_tool_calls": round(statistics.mean(len(r.get("tools", [])) for r in records), 2),
        "node_p50_ms": {k: _pct(v, 0.5) for k, v in sorted(per_node.items())},
        "with_wines": sum(r.get("wines", 0) > 0 for r in records),
        # доля ходов, где отвечала запасная модель: рост — сигнал, что основная деградирует у провайдера
        "llm_fallback_rate": round(sum(any(e["kind"] == "llm_fallback" for e in r.get("events", []))
                                       for r in records) / len(records), 3),
    }
