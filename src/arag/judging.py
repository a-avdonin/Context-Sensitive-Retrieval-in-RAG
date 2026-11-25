"""Калибровка LLM-судьи на human-оценках MTRAG и сводка качества ответов по системам."""
import json
import statistics as st
from pathlib import Path

from .analysis import md_table
from .generation import CRITERIA

HUMAN_FILE = "mtrag-human/evaluations/reference_subset_with_human_evaluations.json"
HUMAN_METRICS = ("faithfulness", "completeness", "appropriateness", "naturalness", "win-rate")
OFFICIAL_METRICS = ("rb_llm", "rl_f", "rb_agg")  # автоматические метрики авторов MTRAG, уже посчитаны в файле


def _human_value(ann: dict, metric: str) -> float | None:
    vals = [float(v["value"]) for v in (ann.get(metric) or {}).values() if v.get("value") not in (None, "")]
    if not vals:
        return None
    return st.mean(vals) if metric == "win-rate" else st.median(vals)  # median — агрегатор из файла


def load_human_eval(data_dir: str) -> list[dict]:
    """477 ответов (159 задач × reference / llama-3.1-405b / gpt-4o) с оценками трёх аннотаторов (шкала 1–4)."""
    d = json.load(open(Path(data_dir) / HUMAN_FILE))
    docs = {x["document_id"]: (f"{x.get('title') or ''}\n{x['text']}").strip() for x in d["documents"]}
    tasks = {t["task_id"]: t for t in d["tasks"]}
    items = []
    for e in d["evaluations"]:
        t, ann = tasks[e["task_id"]], e["annotations"]
        items.append({
            "task_id": e["task_id"], "model_id": e["model_id"], "answer": e["model_response"],
            "turns": [{"speaker": x["speaker"], "text": x["text"]} for x in t["input"]],
            "reference": (t.get("targets") or [{"text": ""}])[0]["text"],
            "passages": [docs[c["document_id"]] for c in t.get("contexts", []) if c["document_id"] in docs],
            "answerability": (t.get("Answerability") or [None])[0],
            "human": {m: _human_value(ann, m) for m in HUMAN_METRICS},
            "official": {m: ((ann.get(m) or {}).get("composite") or {}).get("value") for m in OFFICIAL_METRICS},
        })
    return items


def _corr(xs: list, ys: list) -> dict:
    from scipy.stats import kendalltau, spearmanr

    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3 or len({x for x, _ in pairs}) < 2 or len({y for _, y in pairs}) < 2:
        return {"n": len(pairs), "spearman": None, "kendall": None}
    a, b = zip(*pairs)
    return {"n": len(pairs), "spearman": float(spearmanr(a, b)[0]), "kendall": float(kendalltau(a, b)[0])}


def calibration_report(items: list[dict], judge_model: str) -> str:
    """items — результат load_human_eval с добавленным полем judge ({} — судья не вернул оценки)."""
    S = [f"# Калибровка LLM-судьи: `{judge_model}`\n",
         f"Ответов: {len(items)} ({len({i['task_id'] for i in items})} задач × 3 системы). Человеческие оценки — "
         "шкала 1–4, медиана трёх аннотаторов; win-rate — среднее. Судья — шкала 1–5. "
         f"Судья не вернул оценки: {sum(not i['judge'] for i in items)}.\n"]

    subsets = {"все ответы": items, "без reference-ответов": [i for i in items if i["model_id"] != "reference"]}
    for name, sub in subsets.items():
        S.append(f"## Согласие с людьми: {name} (n = {len(sub)})\n")
        S.append("Спирмен ρ (в скобках Кендалл τ). Строки — автоматические оценки, столбцы — оценки людей.\n")
        rows = []
        for src, keys in (("judge", CRITERIA), ("official", OFFICIAL_METRICS)):
            for k in keys:
                row = {"оценка": f"судья: {k}" if src == "judge" else f"MTRAG: {k}"}
                for hm in HUMAN_METRICS:
                    c = _corr([i[src].get(k) for i in sub], [i["human"][hm] for i in sub])
                    row[hm] = "—" if c["spearman"] is None else f"{c['spearman']:.2f} ({c['kendall']:.2f})"
                rows.append(row)
        S.append(md_table(rows, ["оценка", *HUMAN_METRICS]) + "\n")
    S.append("`MTRAG: rb_llm / rl_f / rb_agg` — метрики авторов бенчмарка (GPT-4o-mini-судья по эталону, "
             "RAGAS-faithfulness, алгоритмическая). Это ориентир: насколько вообще автоматические оценки "
             "согласуются с людьми на этих данных. У reference-ответов correctness и completeness тривиальны "
             "(ответ совпадает с эталоном), поэтому главный срез — «без reference-ответов».\n")

    S.append("## Средние по системам\n")
    rows = []
    for m in sorted({i["model_id"] for i in items}):
        sub = [i for i in items if i["model_id"] == m]
        row = {"система": m, "n": len(sub)}
        for hm in ("faithfulness", "completeness", "appropriateness"):
            row[f"люди: {hm}"] = float(st.mean(v for i in sub if (v := i["human"][hm]) is not None))
        for c in CRITERIA:
            vals = [i["judge"][c] for i in sub if i["judge"]]
            row[f"судья: {c}"] = float(st.mean(vals)) if vals else None
        rows.append(row)
    S.append(md_table(rows, list(rows[0].keys())) + "\n")

    S.append("## Распределение оценок судьи\n")
    rows = [{"критерий": c} | {str(v): sum(i["judge"].get(c) == v for i in items) for v in range(1, 6)} for c in CRITERIA]
    S.append(md_table(rows, ["критерий", "1", "2", "3", "4", "5"]) + "\n")
    return "\n".join(S)


def system_name(path: Path) -> str:
    """runs/<ds>_answers/<stem>.jsonl → имя системы, как в `arag significance`."""
    stem = path.stem
    return "route/" + stem[len("route_"):] if stem.startswith("route_") else stem.replace("__", ":")


def answers_table(files: list[Path], keep) -> tuple[list[dict], dict[str, dict[str, dict]]]:
    """Средние оценки судьи по системам. keep(task_id) — фильтр среза. Возвращает строки и per-task оценки."""
    from .metrics import read_jsonl
    from .queries import is_refusal

    rows, per_task = [], {}
    for p in files:
        recs = [r for r in read_jsonl(p) if keep(r["task_id"])]
        if not recs:
            continue
        ok = [r for r in recs if r["judge"]]
        per_task[system_name(p)] = {r["task_id"]: r["judge"] for r in ok}
        row = {"system": system_name(p), "n": len(recs), "judged": len(ok)}
        for c in CRITERIA:
            row[c] = float(st.mean(r["judge"][c] for r in ok)) if ok else None
        row |= {"declined": sum(is_refusal(r["answer"]) for r in recs) / len(recs),
                "gen_tokens": float(st.mean(r["gen_tokens"] for r in recs)),
                "judge_tokens": float(st.mean(r["judge_tokens"] for r in recs))}
        rows.append(row)
    return rows, per_task
