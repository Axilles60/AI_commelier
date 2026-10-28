"""Выгрузка реального каталога «Своё Вино» (vino-svoe.ru) в data/catalog.json.

Источник — публичный API сайта: /api/wines (список) и /api/wines/{slug} (карточка).
Ничего не дополняем: в каталог попадают только поля, которые вернул сайт.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

BASE = "https://vino-svoe.ru"
OUT = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
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


def normalize(slug, d):
    cat = (d.get("category") or {}).get("name", "")
    return {
        "id": slug,
        "title": d.get("title"),
        "url": f"{BASE}/wines/{slug}",
        "category": cat,                      # напр. «Красное сухое»
        "color_name": d.get("color"),         # цвет в бокале, как на сайте
        "manufacturer": (d.get("manufacturer") or {}).get("name"),
        "region": (d.get("region") or {}).get("name"),
        "grapes": [g["name"] for g in d.get("grapes") or []],
        "dishes": [x["name"] for x in d.get("dishes") or []],
        "alcohol": d.get("alcohol"),
        "alcohol_max": d.get("alcoholMax"),
        "temperature": d.get("temperature"),
        "description": d.get("description") or "",
        "rating": d.get("publicRating"),
    }


def main(limit=None):
    slugs, page = [], 1
    while True:
        data = get("/api/wines", page=page, perPage=24)
        if not data:
            break
        slugs += [w["slug"] for w in data["items"]]
        if page >= data["totalPages"] or (limit and len(slugs) >= limit):
            break
        page += 1
    slugs = slugs[:limit] if limit else slugs
    print(f"slugs: {len(slugs)}", flush=True)

    def card(slug):
        d = get(f"/api/wines/{slug}")
        return normalize(slug, d) if d else None

    with ThreadPoolExecutor(max_workers=4) as ex:
        wines = [w for w in ex.map(card, slugs) if w]
    OUT.parent.mkdir(exist_ok=True)
    meta = {"source": BASE, "fetched_at": time.strftime("%Y-%m-%d"), "count": len(wines),
            "note": "Цены и наличие публичный API не отдаёт — агент их не называет."}
    OUT.write_text(json.dumps({"meta": meta, "wines": wines}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(wines)} -> {OUT}")


def diff():
    """Сверка с сайтом без перезаписи: какие вина появились на vino-svoe.ru и какие исчезли."""
    local = {w["id"] for w in json.loads(OUT.read_text(encoding="utf-8"))["wines"]}
    remote, page = set(), 1
    while True:
        data = get("/api/wines", page=page, perPage=24)
        if not data:
            raise SystemExit(f"не удалось получить страницу {page} — сверка прервана")
        remote |= {w["slug"] for w in data["items"]}
        if page >= data["totalPages"]:
            break
        page += 1
    added, removed = sorted(remote - local), sorted(local - remote)
    print(f"на сайте: {len(remote)}, в каталоге: {len(local)}; новых: {len(added)}, исчезнувших: {len(removed)}")
    for title, items in (("новые", added), ("исчезли", removed)):
        if items:
            print(f"  {title}: {', '.join(items[:15])}{' …' if len(items) > 15 else ''}")
    return added, removed


if __name__ == "__main__":
    if "--diff" in sys.argv:
        diff()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
