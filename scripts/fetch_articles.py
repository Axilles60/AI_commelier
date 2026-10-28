"""Выгрузка статей портала «Своё Вино» (vino-svoe.ru/articles) в data/articles.json.

Источник — публичный API сайта: /api/articles (список) и /api/articles/{slug} (текст). Ничего не дополняем:
сохраняем заголовок, категорию, дату, ссылку и текст статьи без HTML-разметки.
"""
import html
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://vino-svoe.ru"
OUT = Path(__file__).resolve().parent.parent / "data" / "articles.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Sommelier prototype)"}


def get(path, **params):
    for attempt in range(4):
        try:
            r = requests.get(BASE + path, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1 + attempt * 2)
    return None


def to_text(fragment: str) -> str:
    text = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", "\n", fragment or "", flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def article(slug: str):
    d = get(f"/api/articles/{slug}")
    if not d:
        return None
    parts = [d.get("description") or ""]
    for block in d.get("blocks") or []:  # у части статей текст разбит на блоки
        for key in ("text", "content", "description", "body"):
            if isinstance(block.get(key), str):
                parts.append(block[key])
    return {"id": slug, "url": f"{BASE}/articles/{slug}", "title": (d.get("title") or "").strip(),
            "category": (d.get("category") or {}).get("name"), "published": (d.get("firstPublishedAt") or "")[:10],
            "text": to_text("\n".join(parts))}


def main(limit=None):
    slugs, page = [], 1
    while True:
        data = get("/api/articles", page=page, perPage=24)
        if not data:
            break
        slugs += [a["slug"] for a in data["items"]]
        if page >= data["totalPages"] or (limit and len(slugs) >= limit):
            break
        page += 1
    slugs = slugs[:limit] if limit else slugs
    print(f"slugs: {len(slugs)}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        items = [a for a in ex.map(article, slugs) if a and len(a["text"]) > 200]
    meta = {"source": f"{BASE}/articles", "fetched_at": time.strftime("%Y-%m-%d"), "count": len(items)}
    OUT.parent.mkdir(exist_ok=True)  # в свежем клоне папки data/ ещё нет
    OUT.write_text(json.dumps({"meta": meta, "articles": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(items)} -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
