import json
from pathlib import Path

import pytrec_eval


def per_query(qrels: dict, run: dict[str, dict[str, float]], k_values=(1, 3, 5, 10)) -> dict[str, dict[str, float]]:
    """Метрики по каждому запросу. Запросы без qrels пропускаются."""
    ks = ",".join(map(str, k_values))
    run = {q: docs for q, docs in run.items() if q in qrels}
    qrels = {q: r for q, r in qrels.items() if q in run}
    ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{ks}", f"recall.{ks}"})
    return ev.evaluate(run)


def aggregate(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    keys = next(iter(scores.values())).keys()
    return {k: sum(s[k] for s in scores.values()) / len(scores) for k in keys}


def records_to_run(records: list[dict]) -> dict[str, dict[str, float]]:
    # pytrec_eval сортирует по score; ранги переводим в убывающие скоры, чтобы не зависеть от шкалы
    return {r["task_id"]: {d: float(len(r["hits"]) - i) for i, (d, _) in enumerate(r["hits"])} for r in records}


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
