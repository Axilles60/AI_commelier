"""Разнообразие выдачи: жадный MMR-реранкинг (Carbonell & Goldstein, 1998).

Без него топ-5 нередко состоит из двух-трёх вин одной винодельни или одного сорта — клиенту полезнее варианты
разного характера. По arXiv 2401.11506 жадный MMR теряет меньше релевантности, чем переранжирование самой LLM.
MMR только переставляет кандидатов, уже прошедших фильтры, поэтому условия запроса (цвет, сладость, блюдо)
не нарушаются.
"""


def similarity(a: dict, b: dict) -> float:
    """Похожесть двух вин по полям каталога: та же винодельня важнее всего, затем сорта и категория."""
    sim = 0.0
    if a.get("manufacturer") and a.get("manufacturer") == b.get("manufacturer"):
        sim += 0.6
    ga, gb = set(a.get("grapes") or []), set(b.get("grapes") or [])
    if ga and gb:
        sim += 0.3 * len(ga & gb) / len(ga | gb)
    if a.get("category") == b.get("category"):
        sim += 0.1
    return sim


def mmr(results: list, k: int, lam: float = 0.75, pool: int = 30) -> list:
    """results — [(score, wine, reasons)], отсортированные по score. Возвращает k элементов:
    argmax λ·score_norm − (1−λ)·max_sim(уже выбранные)."""
    candidates = results[:pool]
    if len(candidates) <= 1 or k <= 1:
        return candidates[:k]
    top, low = candidates[0][0], candidates[-1][0]
    span = (top - low) or 1.0
    norm = {id(r): (r[0] - low) / span for r in candidates}
    chosen = [candidates[0]]
    rest = candidates[1:]
    while rest and len(chosen) < k:
        best = max(rest, key=lambda r: lam * norm[id(r)] - (1 - lam) * max(similarity(r[1], c[1]) for c in chosen))
        chosen.append(best)
        rest.remove(best)
    return chosen
