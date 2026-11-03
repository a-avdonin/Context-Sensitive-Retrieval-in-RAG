import argparse
import csv
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .data import DOMAINS, load_tasks
from .llm import LLM
from .metrics import aggregate, per_query, read_jsonl, records_to_run, write_jsonl
from .pipeline import Engine, Strategy, all_strategies, record_to_dict
from .routers import build_router

COST_WEIGHT = {"query": {"last": 0, "user_concat": 0, "gold_rewrite": 0, "llm_rewrite": 3},
               "retriever": {"bm25": 0, "dense": 1, "hybrid": 1.5}, "rerank": 2}


def cost_rank(strategy: str) -> float:
    s = Strategy.parse(strategy)
    return COST_WEIGHT["query"][s.query] + COST_WEIGHT["retriever"][s.retriever] + COST_WEIGHT["rerank"] * s.rerank


def _llm(cfg):
    c = cfg["llm"]
    return LLM(c["base_url"], c["api_key"], cfg["cache_dir"], c["temperature"])


def _tasks(cfg, args):
    tasks, qrels = load_tasks(cfg["data_dir"], args.dataset, args.domains)
    if not args.all_tasks:
        tasks = [t for t in tasks if t.meta["has_qrels"]]
    return tasks[: args.limit] if args.limit else tasks, qrels


def _run_dir(cfg, dataset) -> Path:
    return Path(cfg["runs_dir"]) / dataset


def _strategies(names: list[str]) -> list[str]:
    out = []
    for n in names:
        if n == "all":
            out += all_strategies()
        elif n == "no_llm":
            out += [s for s in all_strategies() if "llm_rewrite" not in s]
        else:
            out.append(str(Strategy.parse(n)))
    return list(dict.fromkeys(out))


def cmd_run(cfg, args):
    tasks, _ = _tasks(cfg, args)
    strategies = _strategies(args.strategies)
    engine = Engine(cfg, _llm(cfg) if any("llm" in s for s in strategies) else None)
    for s in strategies:
        path = _run_dir(cfg, args.dataset) / f"{s.replace(':', '__')}.jsonl"
        if path.exists() and not args.overwrite:
            print(f"skip {s}: {path} exists")
            continue
        rows = []
        for t in tqdm(sorted(tasks, key=lambda t: t.domain), desc=s):
            rows.append(record_to_dict(engine.run(t, s)) | {"domain": t.domain})
        write_jsonl(path, rows)


def _load_runs(cfg, dataset) -> dict[str, list[dict]]:
    return {p.stem.replace("__", ":"): read_jsonl(p) for p in sorted(_run_dir(cfg, dataset).glob("*.jsonl"))}


def _summ(records: list[dict], scores: dict) -> dict:
    ids = [r["task_id"] for r in records if r["task_id"] in scores]
    keep = {i: scores[i] for i in ids}
    rec = [r for r in records if r["task_id"] in keep]
    n = max(len(rec), 1)
    return aggregate(keep) | {
        "n": len(rec),
        "latency_ms": sum(r["latency_ms"]["total"] for r in rec) / n,
        "llm_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rec) / n,
        "rerank_pairs": sum(r.get("rerank_pairs", 0) for r in rec) / n,
    }


def _print_table(rows: list[dict], cols: list[str]):
    print("\t".join(cols))
    for r in rows:
        print("\t".join(f"{r.get(c, ''):.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols))


def _slices(tasks):
    by = {t.task_id: t for t in tasks}
    return {
        "all": lambda tid: True,
        "first_turn": lambda tid: by[tid].user_turn == 1,
        "later_turns": lambda tid: by[tid].user_turn > 1,
    }, by


def cmd_eval(cfg, args):
    tasks, qrels = _tasks(cfg, args)
    slices, by = _slices(tasks)
    rows = []
    for s, recs in _load_runs(cfg, args.dataset).items():
        recs = [r for r in recs if r["task_id"] in by]
        scores = per_query(qrels, records_to_run(recs), cfg["eval"]["k_values"])
        groups = {"all": recs} if not args.by_domain else defaultdict(list)
        if args.by_domain:
            for r in recs:
                groups[r["domain"]].append(r)
        for g, grecs in groups.items():
            for sl, f in slices.items():
                sub = [r for r in grecs if f(r["task_id"])]
                rows.append({"strategy": s, "group": g, "slice": sl} | _summ(sub, scores))
    cols = ["strategy", "group", "slice", "n", "recall_5", "recall_10", "ndcg_cut_5", "ndcg_cut_10",
            "latency_ms", "llm_tokens", "rerank_pairs"]
    shown = [r for r in rows if args.slice in ("any", r["slice"])]
    _print_table(sorted(shown, key=lambda r: (r["group"], r["slice"], -r.get("ndcg_cut_5", 0))), cols)
    out = Path(cfg["runs_dir"]) / f"{args.dataset}_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")


def cmd_oracle(cfg, args):
    tasks, qrels = _tasks(cfg, args)
    by = {t.task_id: t for t in tasks}
    runs = _load_runs(cfg, args.dataset)
    metric = cfg["eval"]["oracle_metric"]
    per_strategy = {s: per_query(qrels, records_to_run([r for r in recs if r["task_id"] in by]),
                                 cfg["eval"]["k_values"]) for s, recs in runs.items()}
    order = {s: i for i, s in enumerate(all_strategies())}
    labels, best_scores = [], []
    for tid in by:
        cands = [(sc[tid][metric], -cost_rank(s), -order.get(s, 0), s) for s, sc in per_strategy.items() if tid in sc]
        if not cands:
            continue
        best = max(cands)
        labels.append({"task_id": tid, "domain": by[tid].domain, "best": best[3], metric: best[0],
                       "solvable": best[0] > 0,
                       "scores": {s: sc[tid][metric] for s, sc in per_strategy.items() if tid in sc}})
        best_scores.append(best[0])
    write_jsonl(Path(cfg["runs_dir"]) / f"{args.dataset}_oracle.jsonl", labels)

    unsolved = sum(not l["solvable"] for l in labels)
    print(f"oracle {metric}: {sum(best_scores) / len(best_scores):.4f} over {len(best_scores)} queries "
          f"({unsolved} with zero score for every strategy)")
    for s, sc in sorted(per_strategy.items(), key=lambda x: -aggregate(x[1]).get(metric, 0)):
        wins = sum(l["best"] == s and l["solvable"] for l in labels)
        print(f"{s:32s} {metric}={aggregate(sc)[metric]:.4f}  oracle_picks={wins}")


def cmd_route(cfg, args):
    tasks, qrels = _tasks(cfg, args)
    runs = {s: {r["task_id"]: r for r in recs} for s, recs in _load_runs(cfg, args.dataset).items()}
    llm = _llm(cfg) if args.router.startswith("llm") else None
    router = build_router(args.router, cfg, llm, args.fixed)
    engine = None
    out = []
    for t in tqdm(tasks, desc=args.router):
        route = router.route(t)
        rec = runs.get(route.strategy, {}).get(t.task_id)
        if rec is None:
            engine = engine or Engine(cfg, llm or _llm(cfg))
            rec = record_to_dict(engine.run(t, route.strategy)) | {"domain": t.domain}
        rec = dict(rec)
        rec["latency_ms"] = rec["latency_ms"] | {"router": route.latency_ms,
                                                 "total": rec["latency_ms"]["total"] + route.latency_ms}
        rec["prompt_tokens"] += route.prompt_tokens
        rec["completion_tokens"] += route.completion_tokens
        out.append(rec | {"route": route.strategy, "rationale": route.rationale})

    name = args.router if args.router != "fixed" else f"fixed_{args.fixed.replace(':', '__')}"
    write_jsonl(Path(cfg["runs_dir"]) / f"{args.dataset}_routes" / f"{name}.jsonl", out)
    scores = per_query(qrels, records_to_run(out), cfg["eval"]["k_values"])
    slices, _ = _slices(tasks)
    rows = [{"router": name, "slice": sl} | _summ([r for r in out if f(r["task_id"])], scores)
            for sl, f in slices.items()]
    _print_table(rows, ["router", "slice", "n", "recall_5", "ndcg_cut_5", "ndcg_cut_10", "latency_ms", "llm_tokens"])
    dist = defaultdict(int)
    for r in out:
        dist[r["route"]] += 1
    print("\nroutes:", dict(sorted(dist.items(), key=lambda x: -x[1])))


def cmd_answer(cfg, args):
    from .data import load_corpus
    from .generation import generate, judge

    tasks, _ = _tasks(cfg, args)
    by = {t.task_id: t for t in tasks}
    src = Path(args.run)
    recs = [r for r in read_jsonl(src) if r["task_id"] in by]
    llm, c = _llm(cfg), cfg["llm"]
    out = []
    for r in tqdm(recs, desc="answer+judge"):
        t = by[r["task_id"]]
        corpus = load_corpus(cfg["data_dir"], t.domain)
        docs = [d for d, _ in r["hits"][: args.k]]
        answer, g = generate(llm, c["generator_model"], t, docs, corpus, c["max_history_turns"])
        scores, j = judge(llm, c["judge_model"], t, answer, docs, corpus, c["max_history_turns"])
        out.append({"task_id": t.task_id, "answer": answer, "judge": scores,
                    "gen_tokens": g.prompt_tokens + g.completion_tokens, "gen_latency_ms": g.latency_ms,
                    "answerability": t.meta["answerability"]})
    write_jsonl(src.with_name(src.stem + "_answers.jsonl"), out)
    for crit in ("correctness", "faithfulness", "completeness"):
        vals = [o["judge"].get(crit) for o in out if isinstance(o["judge"].get(crit), (int, float))]
        print(f"{crit}: {sum(vals) / max(len(vals), 1):.3f} (n={len(vals)})")


def main():
    p = argparse.ArgumentParser("arag")
    p.add_argument("--config")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", default="mtrag", choices=["mtrag", "mtragun"])
        sp.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
        sp.add_argument("--limit", type=int)
        sp.add_argument("--all-tasks", action="store_true", help="включить задачи без qrels")

    sp = sub.add_parser("run", help="прогнать стратегии и сохранить runs")
    common(sp)
    sp.add_argument("--strategies", nargs="+", default=["last:bm25"],
                    help="напр. last:bm25 llm_rewrite:hybrid+rerank, или all / no_llm")
    sp.add_argument("--overwrite", action="store_true")

    sp = sub.add_parser("eval", help="таблица метрик по всем сохранённым runs")
    common(sp)
    sp.add_argument("--by-domain", action="store_true")
    sp.add_argument("--slice", default="any", choices=["any", "all", "first_turn", "later_turns"])

    sp = sub.add_parser("oracle", help="лучшая стратегия на каждый запрос и верхняя граница")
    common(sp)

    sp = sub.add_parser("route", help="оценить роутер")
    common(sp)
    sp.add_argument("--router", default="rule", choices=["fixed", "rule", "llm_nohist", "llm_hist"])
    sp.add_argument("--fixed", default="last:hybrid+rerank")

    sp = sub.add_parser("answer", help="генерация ответа и LLM-as-a-Judge по файлу run")
    common(sp)
    sp.add_argument("--run", required=True, help="путь к runs/<dataset>/<strategy>.jsonl")
    sp.add_argument("--k", type=int, default=5)

    args = p.parse_args()
    cfg = load_config(args.config)
    cmds = {"run": cmd_run, "eval": cmd_eval, "oracle": cmd_oracle, "route": cmd_route, "answer": cmd_answer}
    cmds[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
