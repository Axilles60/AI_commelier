"""Сравнение нашего агента (student) с Сомельоном (teacher) на датасете eval/teacher_dataset.jsonl.

Для каждого вопроса датасета агент отвечает в чистом диалоге с пустым профилем, затем считаем:
  * student_grounded  — critic не нашёл выдуманных вин, ссылок и цен (наш главный инвариант);
  * teacher_links_ok  — все ссылки teacher ведут на реальные вина каталога;
  * agreement         — близость стиля подбора к teacher (eval.teacher.compare, 0–1);
  * clarify_*         — кто из двух задал уточняющий вопрос без рекомендаций.

  python -m eval.teacher_dataset_eval            # все вопросы
  python -m eval.teacher_dataset_eval d01,g02    # выборочно
"""
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from eval.collect_teacher import load  # noqa: E402
from eval.teacher import compare  # noqa: E402
from sommelier import catalog, memory  # noqa: E402
from sommelier.agent import build_graph, check_grounding, get_topic_llm  # noqa: E402

HERE = Path(__file__).parent
WINE_URL = re.compile(r"vino-svoe\.ru/wines/([\w-]+)")


def wines_in(text: str) -> list:
    return [i for i in dict.fromkeys(WINE_URL.findall(text)) if catalog.get_wine(i)]


def is_clarify(text: str) -> bool:
    return "?" in text and not WINE_URL.search(text)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ["LLM_FALLBACK_MODEL"] = ""  # меряем основную модель, запасная при 429 подменила бы её
    memory.PROFILES_DIR = Path(tempfile.mkdtemp())
    rows = load()
    if not rows:  # иначе пустой прогон перезаписал бы готовый отчёт
        sys.exit("Нет eval/teacher_dataset.jsonl: ответы Сомельона — тексты vino-svoe.ru, в репозиторий они не входят. "
                 "Соберите их сами (eval/collect_teacher.py, вопросы — eval/teacher_queries.json) или смотрите готовый "
                 "результат в eval/teacher_dataset_report.json и README.")
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    rows = {k: v for k, v in rows.items() if not only or k in only}
    graph = build_graph(topic_llm=get_topic_llm())
    out, by_scen = [], defaultdict(list)
    for qid, r in rows.items():
        cfg = {"configurable": {"thread_id": f"ds_{qid}", "user_id": f"ds_{qid}"}}
        res = graph.invoke({"messages": [HumanMessage(content=r["query"])]}, config=cfg)
        answer = res["messages"][-1].content
        s_ids = wines_in(answer)
        row = {
            "id": qid, "scenario": r["scenario"], "query": r["query"],
            "student_grounded": not check_grounding(answer, res["messages"]),
            "teacher_links_ok": not any(m.startswith("/wines/") for m in r["not_in_catalog"]),
            "agreement": compare(s_ids, r["teacher_wine_ids"])["score"],
            "student_clarify": is_clarify(answer), "teacher_clarify": is_clarify(r["teacher_text"]),
            "student_wines": s_ids, "teacher_wines": r["teacher_wine_ids"], "student_answer": answer,
        }
        out.append(row)
        by_scen[r["scenario"]].append(row)
        print(f"{qid:4} {r['scenario']:7} grounded={'✓' if row['student_grounded'] else '✗'} "
              f"teacher_links={'✓' if row['teacher_links_ok'] else '✗'} agreement={row['agreement']} "
              f"clarify s/t={int(row['student_clarify'])}/{int(row['teacher_clarify'])}", flush=True)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return f"{sum(xs) / len(xs):.2f} (n={len(xs)})" if xs else "—"

    print("\nсценарий  | student grounded | teacher links ok | agreement")
    for scen, rs in sorted(by_scen.items()):
        print(f"{scen:9} | {sum(r['student_grounded'] for r in rs)}/{len(rs):<14} | "
              f"{sum(r['teacher_links_ok'] for r in rs)}/{len(rs):<14} | {avg(r['agreement'] for r in rs)}")
    n = len(out)
    print(f"{'ИТОГО':9} | {sum(r['student_grounded'] for r in out)}/{n:<14} | "
          f"{sum(r['teacher_links_ok'] for r in out)}/{n:<14} | {avg(r['agreement'] for r in out)}")
    path = HERE / "teacher_dataset_report.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report → {path}")


if __name__ == "__main__":
    main()
