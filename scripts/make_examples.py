"""Собирает docs/examples.md — реальные ответы агента из отчётов финального прогона, без правок.

Для проверяющих без ключа OpenRouter: видно, что агент отвечает, какие инструменты вызывает и как это оценили
грейдеры. Источники: eval/report.json (python -m eval.run_eval) и eval/user_sim_report.json (python -m eval.user_sim).
Запуск: python scripts/make_examples.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.user_sim import PERSONAS  # noqa: E402
from sommelier.agent import leaks_secret, load_env  # noqa: E402

GROUPS = [
    ("Подбор по блюду, сорту, региону, поводу", ["t01", "t02", "t03", "t04", "t14", "t15", "t21", "t26", "t27"]),
    ("Уточняющие вопросы и диалог в несколько ходов", ["t05", "t06", "t07", "t25"]),
    ("Персонализация: профиль вкусов", ["t11", "t12", "t16"]),
    ("Честность: цены, несуществующие вина", ["t08", "t09", "t10"]),
    ("Вопросы о вине — по статьям портала", ["t22", "t23"]),
    ("Только тема вина и защита секрета", ["t13", "t17", "t18", "t19", "t20", "t24"]),
]
GRADER_RU = {"grounding_grader": "обоснованность", "constraint_grader": "условия запроса",
             "behavior_grader": "поведение", "state_grader": "профиль"}


def quote(text: str) -> str:
    return "\n".join("> " + line if line.strip() else ">" for line in text.strip().splitlines())


def args_short(call: dict) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.get("args", {}).items() if k != "limit")
    return f"`{call['tool']}({args})`"


def main():
    load_env()
    secret = os.getenv("SOMMELIER_SECRET", "")
    aliases = [a for a in os.getenv("SOMMELIER_SECRET_ALIASES", "").split(",") if a]
    tasks = {t["id"]: t for t in json.loads((ROOT / "eval/tasks.json").read_text(encoding="utf-8"))}
    report = json.loads((ROOT / "eval/report.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in report["rows"]}
    sims = json.loads((ROOT / "eval/user_sim_report.json").read_text(encoding="utf-8"))
    goals = {p["id"]: p["goal"] for p in PERSONAS}

    out = ["# Примеры ответов агента",
           "",
           "Реальные ответы Qwen 3.8 Flash из финального прогона версии v1.0, **без правок**. Файл собран скриптом",
           "`python scripts/make_examples.py` из `eval/report.json` и `eval/user_sim_report.json`, так что его можно",
           "пересобрать после своего прогона. Ссылки ведут на настоящие карточки и статьи vino-svoe.ru.",
           "",
           f"Итог прогона: {sum(r['pass'] for r in report['rows'])}/{report['n']} задач прошли все грейдеры "
           "(обоснованность, условия запроса, поведение, профиль).",
           ""]
    for title, ids in GROUPS:
        out += [f"## {title}", ""]
        for tid in ids:
            t, r = tasks[tid], rows.get(tid)
            if not r:
                continue
            turns = [t["query"]] + t.get("followups", [])
            out.append(f"### {tid}. " + " → ".join(f"«{x}»" for x in turns))
            if t.get("profile"):
                out.append(f"Профиль пользователя до диалога: `{json.dumps(t['profile'], ensure_ascii=False)}`")
            calls = r.get("tool_calls") or [{"tool": x, "args": {}} for x in r["tools"]]
            out.append("Инструменты: " + (", ".join(args_short(c) for c in calls) if calls else "не вызывались"))
            grades = ", ".join(f"{GRADER_RU[k]} {'✓' if v['pass'] else '✗'}" for k, v in r["grades"].items())
            out += [f"Оценка: {grades}" + (f"; ход {r['seconds']} с" if r.get("seconds") else ""), "",
                    quote(r["answer"]), ""]
    out += ["## Диалоги с симулятором покупателя", "",
            "Покупателя играет LLM со скрытой целью (`eval/user_sim.py`); агент должен сам выяснить её вопросами.",
            "Если подборка не подходит, покупатель говорит, что не так. Показан первый прогон каждого покупателя.", ""]
    for s in sims:
        if s["run"] != 0:
            continue
        verdict = "подходящая подборка" if s["success"] else "цель не достигнута"
        out += [f"### {s['id']}. Цель покупателя: {goals.get(s['id'], '')}",
                f"Итог: {verdict}, ходов: {s.get('turns') or len(s['dialog']) // 2}; "
                f"с первой подборки: {'да' if s.get('first_rec_ok') else 'нет'}", ""]
        for who, text in s["dialog"]:
            out += [f"**{who}:**", "", quote(text), ""]
    text = "\n".join(out) + "\n"
    if secret and leaks_secret(text, secret, aliases):
        sys.exit("В примерах встретилось секретное слово — файл не записан")
    path = ROOT / "docs" / "examples.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"{path} — {len(text.splitlines())} строк")


if __name__ == "__main__":
    main()
