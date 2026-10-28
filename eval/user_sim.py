"""Симулятор пользователя (как в τ-bench): покупателя играет LLM со скрытой целью.

В run_eval реплики пользователя заданы заранее («Iron User»): агент не может «промахнуться» с уточняющим вопросом —
следующая реплика всё равно та же. Здесь покупатель отвечает только на то, о чём его спросили, и не выкладывает
всё сразу. Агенту нужно самому выяснить цель уточняющими вопросами и подобрать вина, которые ей удовлетворяют.

Метрики: успех (за MAX_TURNS ходов подборка удовлетворила скрытым условиям; неподходящую покупатель, как живой,
поправляет), успех с первой подборки (строже), число ходов, лишние вопросы (больше двух уточнений), время хода.
Запуск: python -m eval.user_sim [--repeat 2] [--only u8,u10]
"""
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from eval.run_eval import Trajectory, constraint_grader, recommended  # noqa: E402
from sommelier import memory, tracing  # noqa: E402
from sommelier.agent import (_current_turn, _text, budget_claims, build_graph, chat, get_llm,  # noqa: E402
                             get_topic_llm)

MAX_TURNS = 4
PRICE = re.compile(r"\d[\d\s]*(₽|руб)")
PERSONAS = [
    {"id": "u1", "opening": "Хочу вино",
     "goal": "Вино к стейку на ужин. Тебе нравится красное сухое.",
     "constraints": {"color": "красное", "sweetness_any": ["сухое"], "dish_any": ["Мясо и стейки"]}},
    {"id": "u2", "opening": "Нужен подарок",
     "goal": "Подарок отцу на юбилей. Отец пьёт только крымские вина, любит красное полусладкое.",
     "constraints": {"color": "красное", "sweetness_any": ["полусладкое"], "region": "Крым"}},
    {"id": "u3", "opening": "Подобрать вино к блюду",
     "goal": "На ужин будет запечённая форель. Хочешь белое и точно не сладкое.",
     "constraints": {"color": "белое", "not_sweetness_any": ["сладкое", "полусладкое"],
                     "dish_any": ["Блюда из рыбы", "Рыба и морепродукты"]}},
    {"id": "u4", "opening": "Вино на праздник",
     "goal": "Новый год, к закускам. Хочешь игристое брют.",
     "constraints": {"sparkling": True, "sweetness_any": ["брют", "экстра брют"]}},
    {"id": "u5", "opening": "Подобрать вино по вкусу",
     "goal": "Любишь лёгкие фруктовые белые вина, особенно Рислинг, полусухие.",
     "constraints": {"color": "белое", "grape": "рислинг", "sweetness_any": ["полусухое"]}},
    {"id": "u6", "opening": "Вино из определённого региона",
     "goal": "Интересует Долина Дона, хочешь попробовать местный сорт Красностоп.",
     "constraints": {"region": "Долина Дона", "grape": "красностоп"}},
    # посложнее: запреты, выбор из двух цветов, бюджет (цен в каталоге нет), редкий автохтонный сорт
    {"id": "u7", "opening": "Посоветуй вино",
     "goal": "К запечённой утке. Терпеть не можешь сладкое и полусладкое. Цвет без разницы.",
     "constraints": {"dish_any": ["Блюда из птицы"], "not_sweetness_any": ["сладкое", "полусладкое"]}},
    {"id": "u8", "opening": "Нужно вино на свидание",
     "goal": "Будут суши. Хочешь розовое или белое, только не сладкое.",
     "constraints": {"color_any": ["розовое", "белое"], "not_sweetness_any": ["сладкое", "полусладкое"],
                     "dish_any": ["Азиатская кухня", "Рыба и морепродукты"]}},
    {"id": "u9", "opening": "Хочу что-то необычное",
     "goal": "Хочешь попробовать редкий донской сорт Цимлянский чёрный, красное вино.",
     "constraints": {"grape": "цимлянск"}},
    {"id": "u10", "opening": "Вино к пасте до 1000 рублей",
     "goal": "Паста с томатным соусом, хочешь красное. Бюджет до 1000 рублей.",
     "constraints": {"color": "красное", "dish_any": ["Паста"]}, "no_price": True},
    {"id": "u11", "opening": "Подобрать вино по вкусу",
     "goal": "Любишь красное сухое, но терпеть не можешь Каберне Совиньон.",
     "constraints": {"color": "красное", "sweetness_any": ["сухое"], "not_grape": "каберне совиньон"}},
]

# v1 «называй только то, о чём спросили» оказался слишком скрытным: на «есть пожелания по цвету?» покупатель
# отвечал «запечённая форель», умалчивая «белое, не сладкое», — 5 из 7 провалов были ошибкой симулятора, а не агента.
# Теперь он немногословный, но сотрудничающий: на вопрос о пожеланиях называет все свои обязательные условия.
SIM_PROMPT = """Ты покупатель в чате винного портала. Твоя цель: {goal}
Правила: отвечай как живой человек, коротко (одна-две фразы), по-русски. Сам ничего не рассказываешь, но если
сомелье спрашивает о блюде, поводе, вкусах или пожеланиях — назови ВСЕ условия из цели, которые к этому относятся
(цвет, сладость, сорт, регион, блюдо). О том, чего нет в цели, отвечай «без разницы».
Не упоминай, что ты симулятор, и не задавай встречных вопросов."""


def grade_reply(persona: dict, reply: str, history: list) -> dict:
    t = Trajectory(persona["id"], final_response=reply)
    grade = constraint_grader(t, persona)
    # цен в каталоге нет: сумма в рублях, которой не называл сам покупатель (его бюджет), выдумана
    said = {re.sub(r"\D", "", m.group(0)) for who, text in history if who == "Покупатель"
            for m in re.finditer(r"\d[\d\s]*", text)}
    invented = [m.group(0) for m in PRICE.finditer(reply) if re.sub(r"\D", "", m.group(0)) not in said]
    if persona.get("no_price") and (invented or budget_claims(reply)):
        grade = {"pass": False, "details": grade["details"] + [f"цена или «в ваш бюджет» без данных: "
                                                               f"{invented or budget_claims(reply)}"]}
    return grade


def simulate(graph, sim_llm, persona: dict, run: int) -> dict:
    """Диалог до подходящей подборки (не больше MAX_TURNS ходов). Неподходящая подборка — не конец: покупатель,
    как живой, говорит, что не так («а есть красное?»), и агент получает шанс исправиться (как в τ-bench)."""
    uid = f"sim_{persona['id']}_{run}"
    user_msg, history, times, tools = persona["opening"], [], [], []
    clarifications, rec_grades = 0, []
    for turn in range(1, MAX_TURNS + 1):
        start = time.perf_counter()
        reply = chat(graph, user_msg, uid, "t")
        times.append(time.perf_counter() - start)
        # какие поиски (с аргументами) стояли за ответом этого хода — из состояния графа
        turn_msgs = _current_turn(graph.get_state({"configurable": {"thread_id": f"{uid}:t"}}).values["messages"])
        tools.append([{"name": c["name"], "args": c["args"]} for m in turn_msgs for c in (getattr(m, "tool_calls", None) or [])])
        history += [("Покупатель", user_msg), ("Сомелье", reply)]
        hint = ""
        if recommended(Trajectory(persona["id"], final_response=reply)):
            grade = grade_reply(persona, reply, history)
            rec_grades.append(grade)
            if grade["pass"]:
                break
            hint = ("\nСомелье предложил вина, но они не подходят под твою цель. Коротко скажи, что не так "
                    "(как покупатель, своими словами).")
        else:
            clarifications += 1
        dialog = "\n".join(f"{who}: {text}" for who, text in history)
        user_msg = _text(sim_llm.invoke([SystemMessage(content=SIM_PROMPT.format(goal=persona["goal"]) + hint),
                                         HumanMessage(content=f"Диалог:\n{dialog}\n\nТвоя следующая реплика:")])).strip()
    success = bool(rec_grades) and rec_grades[-1]["pass"]
    return {"id": persona["id"], "run": run, "success": success, "turns": len(times) if success else None,
            "first_rec_ok": bool(rec_grades) and rec_grades[0]["pass"], "recommendations": len(rec_grades),
            "clarifications": clarifications,
            "details": rec_grades[-1]["details"] if rec_grades else [f"за {MAX_TURNS} хода нет рекомендации"],
            "dialog": history, "tools": tools, "turn_seconds": [round(x, 1) for x in times]}


def main():
    os.environ["LLM_FALLBACK_MODEL"] = ""  # меряем основную модель
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    memory.PROFILES_DIR, tracing.TRACES_DIR = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    graph, sim_llm = build_graph(topic_llm=get_topic_llm()), get_llm()
    rows = []
    for run in range(repeat):
        for p in (x for x in PERSONAS if not only or x["id"] in only):
            try:
                r = simulate(graph, sim_llm, p, run)
            except Exception as e:  # сбой провайдера не роняет прогон
                r = {"id": p["id"], "run": run, "success": False, "error": f"{type(e).__name__}: {e}"[:200]}
            rows.append(r)
            print(f"{'✅' if r['success'] else '❌'} {p['id']} r{run} ходов={r.get('turns')} "
                  f"с первой подборки={'да' if r.get('first_rec_ok') else 'нет'} "
                  f"уточнений={r.get('clarifications')} {r.get('details', r.get('error'))}", flush=True)
    ok = [r for r in rows if r["success"]]
    turns = [r["turns"] for r in rows if r.get("turns")]
    secs = [s for r in rows for s in r.get("turn_seconds", [])]
    print(f"\nуспех за {MAX_TURNS} хода: {len(ok)}/{len(rows)}; с первой подборки: "
          f"{sum(1 for r in rows if r.get('first_rec_ok'))}/{len(rows)}; ходов до подходящей подборки: "
          f"{statistics.mean(turns):.1f}" if turns else "")
    print(f"больше двух уточнений: {sum(1 for r in rows if r.get('clarifications', 0) > 2)}; "
          f"время хода: среднее {statistics.mean(secs):.1f} с" if secs else "")
    out = Path(__file__).parent / ("user_sim_partial.json" if only else "user_sim_report.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report → {out}")


if __name__ == "__main__":
    main()
