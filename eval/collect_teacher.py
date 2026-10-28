"""Сохранение ответа Сомельона в датасет teacher (eval/teacher_dataset.jsonl).

Ответы собираются через обычный виджет на сайте во встроенном браузере: один вопрос в чистом
диалоге, человеческий темп, остановка при лимите или проверке на бота. Скрипт только записывает
то, что показал виджет, и сразу сопоставляет ссылки с каталогом.

  python -m eval.collect_teacher <id> < {"q": "...", "text": "...", "urls": [...]}
  python -m eval.collect_teacher stats
"""
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.teacher import extract_wines  # noqa: E402

HERE = Path(__file__).parent
DATASET = HERE / "teacher_dataset.jsonl"
QUERIES = {q["id"]: q for q in json.loads((HERE / "teacher_queries.json").read_text(encoding="utf-8"))}


def load() -> dict:
    if not DATASET.exists():
        return {}
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def save(qid: str, payload: dict) -> dict:
    q = QUERIES[qid]
    if payload.get("q", "").strip() != q["query"]:
        raise SystemExit(f"{qid}: в виджете последний вопрос «{payload.get('q')}», а не «{q['query']}» — не сохраняю")
    text = payload["text"] + ("\n\nСсылки: " + " ".join(payload["urls"]) if payload["urls"] else "")
    ex = extract_wines(text)
    row = {**q, "teacher_text": text, "teacher_wine_ids": ex["ids"], "not_in_catalog": ex["not_in_catalog"],
           "collected_at": datetime.date.today().isoformat(), "source": "vino-svoe.ru · Сомельон (виджет на сайте)"}
    rows = load()
    rows[qid] = row
    DATASET.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows.values()) + "\n", encoding="utf-8")
    return row


def verify():
    """Ссылки teacher, которых нет в выгрузке, проверяем по живому публичному API карточки:
    200 — каталог устарел, иначе — битая ссылка у самого помощника."""
    import time
    import requests
    rows = load()
    for r in rows.values():
        slugs = [m.removeprefix("/wines/") for m in r["not_in_catalog"] if m.startswith("/wines/")]
        r["broken_links_verified"] = []
        for s in slugs:
            code = requests.get(f"https://vino-svoe.ru/api/wines/{s}", timeout=30,
                                headers={"User-Agent": "Mozilla/5.0 (AI-Sommelier prototype)"}).status_code
            if code != 200:
                r["broken_links_verified"].append(s)
            time.sleep(1)
    DATASET.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows.values()) + "\n", encoding="utf-8")
    broken = {k: r["broken_links_verified"] for k, r in rows.items() if r["broken_links_verified"]}
    print(f"проверено по живому API; битых ссылок: {sum(map(len, broken.values()))} в {len(broken)} ответах: {broken}")


def stats():
    rows = load()
    print(f"собрано {len(rows)}/{len(QUERIES)}")
    print("по сценариям:", dict(Counter(r["scenario"] for r in rows.values())))
    with_wines = [r for r in rows.values() if r["teacher_wine_ids"]]
    print(f"с рекомендациями из каталога: {len(with_wines)}, вин всего: {sum(len(r['teacher_wine_ids']) for r in with_wines)}")
    broken = {r["id"]: r["not_in_catalog"] for r in rows.values() if any(m.startswith("/wines/") for m in r["not_in_catalog"])}
    print("битые ссылки:", broken or "нет")
    missing = [k for k in QUERIES if k not in rows]
    print("не собрано:", missing or "—")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    if sys.argv[1] in ("stats", "verify"):
        {"stats": stats, "verify": verify}[sys.argv[1]]()
    else:
        payload = json.load(sys.stdin)
        if isinstance(payload, str):  # вывод JSON.stringify со страницы вставлен как есть
            payload = json.loads(payload, strict=False)
        r = save(sys.argv[1], payload)
        print(f"saved {r['id']}: вин {len(r['teacher_wine_ids'])}" + (f", НЕ в каталоге: {r['not_in_catalog']}" if r["not_in_catalog"] else ""))
