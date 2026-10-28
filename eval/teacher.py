"""Teacher: цифровой помощник vino-svoe.ru как эталон для оценки нашего агента (student).

Ответы помощника собираются вручную (виджет на сайте защищён антиботом, автоматически
мы его не опрашиваем) и кладутся в eval/teacher_answers.json:
  [{"id": "t01", "query": "...", "teacher_text": "<ответ помощника как есть>"}]

Что даёт teacher (термины Семинара 4):
  * reference-based grading — сравниваем рекомендации student с эталонными по вину и атрибутам;
  * LLM-as-Judge с эталоном — попарное сравнение ответов (опционально, --judge);
  * аудит самого teacher — какие вина он называет, а в каталоге их нет.

  python -m eval.teacher template   # создать шаблон teacher_answers.json из tasks.json
  python -m eval.teacher audit      # разобрать ответы teacher в позиции каталога
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sommelier import catalog  # noqa: E402

HERE = Path(__file__).parent
TEACHER_PATH = HERE / "teacher_answers.json"
QUOTED = re.compile(r"[«\"“]([^»\"”]{3,120})[»\"”]")


def _title_index():
    """Нормализованные названия → id. Короткие/неоднозначные названия («135», «Пино Нуар»)
    ловим только в кавычках, чтобы не путать с упоминанием сорта или числа."""
    grapes = {catalog.norm(g) for g in catalog.filter_values("grapes")}
    idx, ambiguous = {}, set()
    for w in catalog.load_catalog()["wines"]:
        t = catalog.norm(w["title"]).strip()
        if t in idx and idx[t] != w["id"]:
            ambiguous.add(t)
        idx.setdefault(t, w["id"])
    loose = {t: i for t, i in idx.items() if len(t) >= 8 and not t.isdigit() and t not in grapes and t not in ambiguous}
    return idx, loose


def extract_wines(text: str) -> dict:
    """Ответ teacher → {'ids': [...позиции каталога...], 'not_in_catalog': [«названия», которых нет]}."""
    idx, loose = _title_index()
    ntext = catalog.norm(text)
    slugs = list(dict.fromkeys(re.findall(r"vino-svoe\.ru/wines/([\w-]+)", text)))
    if slugs:  # помощник ставит ссылки на карточки — это точнее, чем угадывать по названиям
        return {"ids": [s for s in slugs if catalog.get_wine(s)],
                "not_in_catalog": [f"/wines/{s}" for s in slugs if not catalog.get_wine(s)]}
    ids = []
    known = {catalog.norm(v) for f in ("category", "region", "manufacturer", "grapes", "dishes")
             for v in catalog.filter_values(f)}
    missing = []
    for q in QUOTED.findall(text):
        nq = catalog.norm(q).strip()
        if nq in idx:
            ids.append(idx[nq])
        elif nq not in known and nq not in ("своё вино", "свое вино"):
            missing.append(q)
    for t in sorted(loose, key=len, reverse=True):  # длинные названия без кавычек
        if t in ntext:
            ids.append(loose[t])
            ntext = ntext.replace(t, " ")
    return {"ids": list(dict.fromkeys(ids)), "not_in_catalog": missing}


def attributes(ids) -> dict:
    ws = [catalog.get_wine(i) for i in ids if catalog.get_wine(i)]
    return {"color": {w["_color"] for w in ws}, "sweetness": {w["_sweet"] for w in ws},
            "region": {w["region"] for w in ws if w.get("region")},
            "grapes": {g for w in ws for g in w["grapes"]}}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def compare(student_ids, teacher_ids) -> dict:
    """Reference-based метрика: совпадение вин + совпадение «стиля» рекомендаций по атрибутам каталога."""
    if not teacher_ids:
        return {"score": None, "details": "teacher не назвал вин из каталога"}
    s, t = attributes(student_ids), attributes(teacher_ids)
    parts = {
        "wine_recall": len(set(student_ids) & set(teacher_ids)) / len(teacher_ids),
        "color": jaccard(s["color"], t["color"]),
        "sweetness": jaccard(s["sweetness"], t["sweetness"]),
        "region": jaccard(s["region"], t["region"]),
        "grapes": jaccard(s["grapes"], t["grapes"]),
    }
    # совпадение конкретных бутылок редко (в каталоге 2250 вин), поэтому основной вес — стиль
    w = {"wine_recall": 0.2, "color": 0.3, "sweetness": 0.2, "region": 0.15, "grapes": 0.15}
    return {"score": round(sum(parts[k] * w[k] for k in w), 3), "details": {k: round(v, 2) for k, v in parts.items()}}


def load_teacher() -> dict:
    if not TEACHER_PATH.exists():
        return {}
    rows = json.loads(TEACHER_PATH.read_text(encoding="utf-8"))
    return {r["id"]: r for r in rows if r.get("teacher_text", "").strip()}


JUDGE_PROMPT = """Ты эксперт-сомелье и оцениваешь ответ ИИ-помощника (STUDENT) по эталону (TEACHER) —
ответу официального помощника портала «Своё Вино» на тот же запрос.
Оцени STUDENT по шкале 1–5 по каждому критерию:
- relevance: насколько рекомендации соответствуют запросу (блюдо, стиль, повод);
- agreement: насколько выбор STUDENT согласуется по стилю с экспертным выбором TEACHER
  (совпадение конкретных бутылок не обязательно);
- explanation: понятно ли объяснён выбор;
- clarification: если запрос размытый — задан ли уместный уточняющий вопрос (иначе 5).
Не снижай оценку за то, что STUDENT не называет цены. Ответь строго JSON:
{"relevance": n, "agreement": n, "explanation": n, "clarification": n, "comment": "..."}"""


def llm_judge(llm, query: str, student: str, teacher: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = llm.invoke([SystemMessage(content=JUDGE_PROMPT),
                       HumanMessage(content=f"Запрос: {query}\n\nTEACHER:\n{teacher}\n\nSTUDENT:\n{student}")])
    m = re.search(r"\{.*\}", resp.content if isinstance(resp.content, str) else str(resp.content), re.S)
    try:
        return json.loads(m.group(0)) if m else {"error": "no json"}
    except json.JSONDecodeError:
        return {"error": "bad json"}


def _template():
    tasks = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
    old = load_teacher()
    rows = [{"id": t["id"], "query": t["query"], "followups": t.get("followups", []),
             "teacher_text": old.get(t["id"], {}).get("teacher_text", "")} for t in tasks]
    TEACHER_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"шаблон → {TEACHER_PATH} ({len(rows)} запросов): вставьте ответы помощника в teacher_text")


def _audit():
    teacher = load_teacher()
    if not teacher:
        print("teacher_answers.json пуст — сначала соберите ответы помощника")
        return
    for tid, r in teacher.items():
        ex = extract_wines(r["teacher_text"])
        print(f"{tid}: в каталоге {len(ex['ids'])} {ex['ids'][:5]}"
              + (f" | НЕ в каталоге: {ex['not_in_catalog']}" if ex["not_in_catalog"] else ""))


def _grade():
    """Те же грейдеры, что и для нашего агента, но на ответах teacher — честное сравнение student vs teacher."""
    from eval.run_eval import TASKS, Step, Trajectory, behavior_grader, constraint_grader
    teacher = load_teacher()
    report_path = HERE / "report.json"
    student = {r["id"]: r for r in json.loads(report_path.read_text(encoding="utf-8"))["rows"]} if report_path.exists() else {}
    tot = {"teacher": [0, 0, 0], "student": [0, 0, 0]}
    print(f"{'id':4} {'запрос':42} | teacher: links constr behav | student: ground constr behav")
    for task in TASKS:
        r = teacher.get(task["id"])
        if not r:
            continue
        t = Trajectory(task["id"], steps=[Step("response", r["teacher_text"])], final_response=r["teacher_text"])
        links_ok = not [m for m in extract_wines(r["teacher_text"])["not_in_catalog"] if m.startswith("/wines/")]
        c, b = constraint_grader(t, task)["pass"], behavior_grader(t, task)["pass"]
        g = student.get(task["id"], {}).get("grades", {})
        sg = [g.get(k, {}).get("pass") for k in ("grounding_grader", "constraint_grader", "behavior_grader")]
        for i, v in enumerate([links_ok, c, b]):
            tot["teacher"][i] += v
        for i, v in enumerate(sg):
            tot["student"][i] += bool(v)
        mark = lambda v: "✓" if v else ("✗" if v is not None else "·")
        print(f"{task['id']:4} {task['query'][:42]:42} |   {mark(links_ok)}     {mark(c)}     {mark(b)}    "
              f"|   {mark(sg[0])}     {mark(sg[1])}     {mark(sg[2])}")
    n = len(teacher)
    print()
    print(f"teacher: валидные ссылки {tot['teacher'][0]}/{n}, условия {tot['teacher'][1]}/{n}, поведение {tot['teacher'][2]}/{n}")
    if student:
        print(f"student: grounding {tot['student'][0]}/{n}, условия {tot['student'][1]}/{n}, поведение {tot['student'][2]}/{n}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"template": _template, "audit": _audit, "grade": _grade}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
