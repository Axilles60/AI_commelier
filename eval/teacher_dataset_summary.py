"""Сводка student vs teacher на датасете Сомельона (после python -m eval.teacher_dataset_eval).

Метрики одинаковые для обоих и считаются по данным карточек каталога:
  * явные условия — если в вопросе назван цвет или сладость, доля рекомендованных вин, которые им соответствуют;
  * ссылки — все ли рекомендованные вина существуют (у teacher битые ссылки проверены по живому API);
  * цена — утверждения о цене/бюджете, которые нельзя проверить: цен в публичных данных нет;
  * уточнение — ответ-вопрос без рекомендаций.

  python -m eval.teacher_dataset_summary
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.collect_teacher import load  # noqa: E402
from sommelier import catalog  # noqa: E402

HERE = Path(__file__).parent
COLOR_STEMS = {"красн": "Красное", "бел": "Белое", "розов": "Розовое", "розе": "Розовое", "оранжев": "Оранжевое"}
PRICE = re.compile(r"\d[\d\s]*\s?(₽|руб)|бюджет", re.I)


def explicit(query: str) -> dict:
    q = catalog.norm(query)
    color = next((v for k, v in COLOR_STEMS.items() if re.search(rf"(^|[^\wё]){k}", q)), None)
    return {"color": color, "sweetness": catalog.sweetness_of(q)}


def compliance(ids: list, cond: dict):
    """Доля вин, подходящих под явные условия вопроса; None — если условий нет или вин нет."""
    if not ids or not (cond["color"] or cond["sweetness"]):
        return None
    ok = 0
    for i in ids:
        w = catalog.get_wine(i)
        ok += bool(w) and (not cond["color"] or w["_color"] == cond["color"]) \
            and (not cond["sweetness"] or w["_sweet"] == cond["sweetness"])
    return ok / len(ids)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    teacher = load()
    if not teacher:  # без ответов Сомельона сводку не пересчитать (они не входят в репозиторий)
        sys.exit("Нет eval/teacher_dataset.jsonl — готовая сводка лежит в eval/teacher_dataset_summary.json")
    student = {r["id"]: r for r in json.loads((HERE / "teacher_dataset_report.json").read_text(encoding="utf-8"))}
    agg = defaultdict(lambda: defaultdict(list))
    rows = []
    for qid, t in teacher.items():
        s = student.get(qid)
        if not s:
            continue
        cond = explicit(t["query"])
        t_links = re.findall(r"vino-svoe\.ru/wines/([\w-]+)", t["teacher_text"])
        row = {
            "id": qid, "scenario": t["scenario"],
            # битые ссылки считаем отдельно (t_links_ok), чтобы не штрафовать за одну ошибку дважды
            "t_compl": compliance(t["teacher_wine_ids"], cond),
            "s_compl": compliance(s["student_wines"], cond),
            "t_links_ok": not t.get("broken_links_verified"), "s_links_ok": s["student_grounded"],
            "t_price": bool(PRICE.search(t["teacher_text"]) and t_links),
            "s_price": bool(re.search(r"\d[\d\s]*\s?(₽|руб)", s["student_answer"])),
            "t_clarify": s["teacher_clarify"], "s_clarify": s["student_clarify"],
            "agreement": s["agreement"],
        }
        rows.append(row)
        for k in ("t_compl", "s_compl", "agreement"):
            if row[k] is not None:
                agg[t["scenario"]][k].append(row[k])
                agg["ВСЕГО"][k].append(row[k])
        for k in ("t_links_ok", "s_links_ok", "t_price", "s_price", "t_clarify", "s_clarify"):
            agg[t["scenario"]][k].append(row[k])
            agg["ВСЕГО"][k].append(row[k])

    def pct(xs):
        return f"{100 * sum(xs) / len(xs):.0f}% (n={len(xs)})" if xs else "—"

    print(f"{'сценарий':9} | {'явные условия: teacher':24} | {'student':16} | {'ссылки ок: teacher':18} | {'student':10} | "
          f"{'цена без данных: t/s':20} | agreement")
    for scen in sorted(k for k in agg if k != "ВСЕГО") + ["ВСЕГО"]:
        a = agg[scen]
        n = len(a["t_links_ok"])
        print(f"{scen:9} | {pct(a['t_compl']):24} | {pct(a['s_compl']):16} | {sum(a['t_links_ok'])}/{n:<16} | "
              f"{sum(a['s_links_ok'])}/{n:<8} | {sum(a['t_price'])}/{sum(a['s_price'])}{'':16} | "
              f"{sum(a['agreement']) / len(a['agreement']):.2f}" if a["agreement"] else "")
    misses = [r for r in rows if r["t_compl"] is not None and r["t_compl"] < 1]
    print("\nгде teacher нарушил явные условия вопроса:", ", ".join(f"{r['id']} ({r['t_compl']:.0%})" for r in misses) or "нигде")
    misses = [r for r in rows if r["s_compl"] is not None and r["s_compl"] < 1]
    print("где student нарушил явные условия вопроса:", ", ".join(f"{r['id']} ({r['s_compl']:.0%})" for r in misses) or "нигде")
    (HERE / "teacher_dataset_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
