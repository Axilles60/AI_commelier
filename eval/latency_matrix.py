"""Матрица A/B по задержке: рассуждения агента (вкл/выкл) × предварительный поиск (вкл/выкл).

Каждая конфигурация — отдельный процесс eval.run_eval на одних и тех же задачах с подбором вин, последовательно
(чтобы конфигурации не делили квоту провайдера). Итог: качество (все грейдеры) и среднее время хода.
  python -m eval.latency_matrix
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TASKS = "t01,t02,t03,t04,t07,t08,t11,t12,t14,t15"
CONFIGS = [("рассуждения вкл, prefetch выкл", "1", "0"), ("рассуждения вкл, prefetch вкл", "1", "1"),
           ("рассуждения выкл, prefetch выкл", "0", "0"), ("рассуждения выкл, prefetch вкл", "0", "1")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    summary = []
    for name, reasoning, prefetch in CONFIGS:
        env = dict(os.environ, AGENT_REASONING=reasoning, SOMMELIER_PREFETCH=prefetch, PYTHONIOENCODING="utf-8")
        tag = f"r{reasoning}p{prefetch}"
        with open(HERE / f"run_matrix_{tag}.log", "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", "-m", "eval.run_eval", "--only", TASKS], cwd=HERE.parent, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
        shutil.copy(HERE / "report_partial.json", HERE / f"report_matrix_{tag}.json")
        rows = json.loads((HERE / f"report_matrix_{tag}.json").read_text(encoding="utf-8"))["rows"]
        ok = [r for r in rows if r.get("seconds") is not None]
        line = {"config": name, "pass": f"{sum(r['pass'] for r in rows)}/{len(rows)}",
                "errors": sum(1 for r in rows if r.get("error")),
                "mean_s": round(statistics.mean(r["seconds"] for r in ok), 1) if ok else None,
                "median_s": round(statistics.median(r["seconds"] for r in ok), 1) if ok else None,
                "searches": round(statistics.mean(r["tools"].count("search_wines") for r in ok), 2) if ok else None,
                "chars": round(statistics.mean(len(r["answer"]) for r in ok)) if ok else None}
        summary.append(line)
        print(json.dumps(line, ensure_ascii=False), flush=True)
    (HERE / "latency_matrix.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
