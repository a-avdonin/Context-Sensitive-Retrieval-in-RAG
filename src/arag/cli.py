import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .config import load_config
from .data import DOMAINS, held_out_dialogues, load_tasks
from .llm import LLM, parallel_map
from .metrics import aggregate, per_query, read_jsonl, records_to_run, write_jsonl
from .pipeline import Engine, Strategy, all_strategies, record_to_dict
from .queries import LLM_QUERY_MODES
from .routers import build_router

COST_WEIGHT = {"query": {"last": 0, "user_concat": 0, "gold_rewrite": 0, "llm_rewrite": 3, "chiq": 6},
               "retriever": {"bm25": 0, "dense": 1, "hybrid": 1.5}, "rerank": 2}


def cost_rank(strategy: str) -> float:
    s = Strategy.parse(strategy)
    return COST_WEIGHT["query"][s.query] + COST_WEIGHT["retriever"][s.retriever] + COST_WEIGHT["rerank"] * s.rerank


def _llm(cfg):
    c = cfg["llm"]
    return LLM(c["base_url"], c["api_key"], cfg["cache_dir"], c["temperature"], c.get("timeout", 120),
               c.get("max_retries", 5), c.get("extra_body"))


def _workers(cfg, args) -> int:
    return args.workers or cfg["llm"].get("workers", 8)


def _prefetch(fn, items, workers: int, desc: str):
    """Параллельно прогреть кэш LLM. Ошибки не роняют прогон: упавшие вызовы повторятся в основном цикле."""
    errors = []

    def safe(x):
        try:
            fn(x)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    parallel_map(safe, items, workers, desc)
    if errors:
        print(f"prefetch {desc}: {len(errors)} ошибок, первая: {errors[0][:300]}")


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
            out += [s for s in all_strategies() if Strategy.parse(s).query not in LLM_QUERY_MODES]
        elif n == "no_rerank":  # всё, что не требует локального cross-encoder
            out += [s for s in all_strategies() if not s.endswith("+rerank")]
        else:
            out.append(str(Strategy.parse(n)))
    return list(dict.fromkeys(out))


def cmd_run(cfg, args):
    tasks, _ = _tasks(cfg, args)
    strategies = _strategies(args.strategies)
    engine = Engine(cfg, _llm(cfg))  # клиент ленивый: к API обращается, только если стратегии это нужно
    for s in strategies:
        path = _run_dir(cfg, args.dataset) / f"{s.replace(':', '__')}.jsonl"
        if path.exists() and not args.overwrite:
            print(f"skip {s}: {path} exists")
            continue
        st = Strategy.parse(s)
        if st.query in LLM_QUERY_MODES:
            _prefetch(lambda t: engine.qb.build(t, st.query), [t for t in tasks if t.history], _workers(cfg, args),
                      f"{st.query} (LLM)")
        if st.retriever in ("dense", "hybrid") and cfg["dense"].get("backend") == "api":
            c = cfg["dense"]
            _prefetch(lambda t: engine.llm.embed_one(c["model"], c.get("query_prefix", "") + engine.query_text(t, st)),
                      tasks, _workers(cfg, args), f"{s}: query embeddings")
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


def is_deployable(strategy: str) -> bool:
    """gold_rewrite использует human rewrite, недоступный при инференсе: только верхняя граница."""
    return Strategy.parse(strategy).query != "gold_rewrite"


def oracle_pick(scores: dict[str, float], eps: float = 0.0) -> tuple[str, float]:
    """Самая дешёвая стратегия с качеством не ниже best - eps. При eps=0 — лучшая, ничьи к дешёвой."""
    order = {s: i for i, s in enumerate(all_strategies())}
    best = max(scores.values())
    ok = [s for s, v in scores.items() if v >= best - eps - 1e-12]
    s = min(ok, key=lambda s: (cost_rank(s), -scores[s], order.get(s, 0)))
    return s, scores[s]


def cmd_oracle(cfg, args):
    tasks, qrels = _tasks(cfg, args)
    by = {t.task_id: t for t in tasks}
    runs = _load_runs(cfg, args.dataset)
    metric = cfg["eval"]["oracle_metric"]
    per_strategy = {s: per_query(qrels, records_to_run([r for r in recs if r["task_id"] in by]),
                                 cfg["eval"]["k_values"]) for s, recs in runs.items()}
    deployable = [s for s in per_strategy if is_deployable(s)]
    if not deployable:
        raise SystemExit("нет runs с реализуемыми стратегиями (все gold_rewrite)")

    labels = []
    for tid in by:
        scores = {s: sc[tid][metric] for s, sc in per_strategy.items() if tid in sc}
        dep = {s: v for s, v in scores.items() if s in deployable}
        if not dep:
            continue
        best, best_v = oracle_pick(dep)
        cheap, cheap_v = oracle_pick(dep, args.eps)
        gold, gold_v = oracle_pick(scores)
        labels.append({"task_id": tid, "domain": by[tid].domain, "user_turn": by[tid].user_turn,
                       "best": best, metric: best_v, "solvable": best_v > 0,
                       "best_cheap": cheap, f"{metric}_cheap": cheap_v,
                       "best_with_gold": gold, f"{metric}_with_gold": gold_v,
                       "scores": scores})
    write_jsonl(Path(cfg["runs_dir"]) / f"{args.dataset}_oracle.jsonl", labels)

    mean = lambda key: sum(l[key] for l in labels) / max(len(labels), 1)
    fixed_best = max(deployable, key=lambda s: aggregate(per_strategy[s]).get(metric, 0))
    print(f"queries: {len(labels)}, unsolvable (0 for every deployable strategy): "
          f"{sum(not l['solvable'] for l in labels)}")
    print(f"best fixed (deployable) {fixed_best:28s} {metric}={aggregate(per_strategy[fixed_best])[metric]:.4f}")
    print(f"oracle (deployable)                           {metric}={mean(metric):.4f}")
    print(f"oracle cheap, eps={args.eps:<4}                        {metric}={mean(f'{metric}_cheap'):.4f}  "
          f"avg cost_rank={sum(cost_rank(l['best_cheap']) for l in labels) / max(len(labels), 1):.2f}")
    print(f"oracle incl. gold_rewrite (upper bound)       {metric}={mean(f'{metric}_with_gold'):.4f}\n")
    for s, sc in sorted(per_strategy.items(), key=lambda x: -aggregate(x[1]).get(metric, 0)):
        wins = sum(l["best"] == s and l["solvable"] for l in labels)
        tag = "" if s in deployable else "  (not deployable)"
        print(f"{s:32s} {metric}={aggregate(sc)[metric]:.4f}  oracle_picks={wins}{tag}")


def cmd_analyze(cfg, args):
    from .analysis import build_report

    tasks, qrels = _tasks(cfg, args)
    by = {t.task_id: t for t in tasks}
    runs = _load_runs(cfg, args.dataset)
    scores = {s: per_query(qrels, records_to_run([r for r in recs if r["task_id"] in by]), cfg["eval"]["k_values"])
              for s, recs in runs.items()}
    report = build_report(tasks, qrels, scores, args.metric, args.dataset)
    out = Path(args.out) if args.out else Path(cfg["runs_dir"]) / f"{args.dataset}_analysis.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"\nsaved {out}")


def cmd_significance(cfg, args):
    import numpy as np

    from .stats import holm, paired_bootstrap, randomization_test

    tasks, qrels = _tasks(cfg, args)
    slices, by = _slices(tasks)
    systems = _load_runs(cfg, args.dataset)
    routes_dir = Path(cfg["runs_dir"]) / f"{args.dataset}_routes"
    systems |= {f"route/{p.stem}": read_jsonl(p) for p in sorted(routes_dir.glob("*.jsonl"))}
    if args.baseline not in systems:
        raise SystemExit(f"нет run для {args.baseline}; есть: {', '.join(systems)}")

    keep = slices[args.slice]
    scores = {s: per_query(qrels, records_to_run([r for r in recs if r["task_id"] in by and keep(r["task_id"])]),
                           cfg["eval"]["k_values"]) for s, recs in systems.items()}
    base = scores[args.baseline]
    rows = []
    for s, sc in scores.items():
        if s == args.baseline:
            continue
        common = sorted(set(base) & set(sc))
        if not common:
            continue
        a = np.array([base[q][args.metric] for q in common])
        b = np.array([sc[q][args.metric] for q in common])
        groups = [by[q].conversation_id for q in common]
        delta, lo, hi = paired_bootstrap(a, b, args.n, groups=groups)
        rows.append({"system": s, "n": len(common), "baseline": a.mean(), "system_score": b.mean(),
                     "delta": delta, "ci_low": lo, "ci_high": hi, "p": randomization_test(a, b, args.n, groups=groups)})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
    print(f"baseline={args.baseline}  metric={args.metric}  slice={args.slice}\n")
    _print_table(sorted(rows, key=lambda r: -r["delta"]),
                 ["system", "n", "baseline", "system_score", "delta", "ci_low", "ci_high", "p", "p_holm"])


def cmd_route(cfg, args):
    train_dataset = args.train_dataset
    if train_dataset and (args.router != "adaptive_rag" or train_dataset == args.dataset):
        raise SystemExit("--train-dataset требует --router adaptive_rag и отдельный датасет обучения")
    tasks, qrels = _tasks(cfg, args)
    runs = {s: {r["task_id"]: r for r in recs} for s, recs in _load_runs(cfg, args.dataset).items()}
    llm = _llm(cfg) if args.router.startswith("llm") or args.router == "adaptive_rag" else None
    router = build_router(args.router, cfg, llm, args.fixed, args.dataset)
    if args.router == "adaptive_rag":
        if train_dataset:
            train_tasks, train_qrels = load_tasks(cfg["data_dir"], train_dataset)
            train_tasks = [t for t in train_tasks if t.meta["has_qrels"]]
            held_out = held_out_dialogues(train_tasks, tasks)
            print(f"Для переноса исключено задач с общим ID или началом диалога: {len(tasks) - len(held_out)}")
            tasks = held_out
            rep = router.prepare_transfer(train_dataset, train_tasks, train_qrels, tasks)
            print(f"adaptive_rag: обучение на {train_dataset}, применение к {args.dataset}")
        else:
            rep = router.prepare(tasks, qrels)
        print(f"{args.router}: уровни {rep['levels']}")
        print(f"  метки (самый простой лучший уровень): {rep['label_dist']}, без метки (все 0): {rep['unlabeled']}")
        print(f"  предсказано: {rep['pred_dist']}")
        if not train_dataset:
            print(f"  CV accuracy {rep['cv_accuracy']:.3f} (всегда частый класс: {rep['majority_accuracy']:.3f})")
            print(f"  confusion (строки — метка): {rep['confusion']}")
    engine = None
    out = []
    # решения роутера считаем параллельно; ошибка API после всех повторов роняет прогон,
    # но уже полученные ответы лежат в кэше — перезапуск продолжит с того же места
    routes = parallel_map(router.route, tasks, _workers(cfg, args) if args.router.startswith("llm") else 1,
                          f"route:{args.router}")
    missing = [t for t, r in zip(tasks, routes)
               if t.task_id not in runs.get(r.strategy, {}) and Strategy.parse(r.strategy).query in LLM_QUERY_MODES]
    if missing:
        engine = Engine(cfg, llm or _llm(cfg))
        for mode in {Strategy.parse(r.strategy).query for t, r in zip(tasks, routes) if t in missing}:
            _prefetch(lambda t: engine.qb.build(t, mode), missing, _workers(cfg, args), f"{mode} (LLM)")
    for t, route in tqdm(list(zip(tasks, routes)), desc=args.router):
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
    if train_dataset:
        name = f"{name}_from_{train_dataset}"
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


def cmd_routes_summary(cfg, args):
    """Сводная таблица по всем роутерам из runs/<dataset>_routes/: качество, решения и стоимость."""
    tasks, qrels = _tasks(cfg, args)
    slices, by = _slices(tasks)
    keep = slices[args.slice]
    rows = []
    for p in sorted((Path(cfg["runs_dir"]) / f"{args.dataset}_routes").glob("*.jsonl")):
        recs = [r for r in read_jsonl(p) if r["task_id"] in by and keep(r["task_id"])]
        if not recs:
            continue
        scores = per_query(qrels, records_to_run(recs), cfg["eval"]["k_values"])
        n = len(recs)
        st = [Strategy.parse(r["route"]) for r in recs]
        later = [s for s, r in zip(st, recs) if by[r["task_id"]].history]
        row = {"router": p.stem, "n": n} | {k: v for k, v in aggregate(scores).items()
                                             if k in ("ndcg_cut_5", "ndcg_cut_10", "recall_5", "recall_10")}
        row |= {
            # rewrite на первом ходе — пустая операция (LLM не вызывается), поэтому считаем долю среди later
            "rewrite_rate": sum(s.query == "llm_rewrite" for s in later) / max(len(later), 1),
            "rerank_rate": sum(s.rerank for s in st) / n,
            "dense_rate": sum(s.retriever == "dense" for s in st) / n,
            "hybrid_rate": sum(s.retriever == "hybrid" for s in st) / n,
            "bm25_rate": sum(s.retriever == "bm25" for s in st) / n,
            "llm_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in recs) / n,
            "router_ms": sum(r["latency_ms"].get("router", 0) for r in recs) / n,
            "latency_ms": sum(r["latency_ms"]["total"] for r in recs) / n,
            "rerank_pairs": sum(r.get("rerank_pairs", 0) for r in recs) / n,
            "parse_errors": sum(r.get("rationale") == "parse_error" for r in recs),
        }
        rows.append(row)
    if not rows:
        raise SystemExit("нет файлов в runs/<dataset>_routes — сначала `arag route ...`")
    cols = ["router", "n", "ndcg_cut_5", "recall_10", "rewrite_rate", "rerank_rate", "dense_rate", "hybrid_rate",
            "bm25_rate", "llm_tokens", "rerank_pairs", "router_ms", "latency_ms", "parse_errors"]
    print(f"slice={args.slice}\n")
    _print_table(sorted(rows, key=lambda r: -r["ndcg_cut_5"]), cols)
    out = Path(cfg["runs_dir"]) / f"{args.dataset}_routes_summary_{args.slice}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")


def _answers_dir(cfg, dataset) -> Path:
    # отдельная папка: в runs/<ds>/ и runs/<ds>_routes/ лежат только run-файлы с hits
    return Path(cfg["runs_dir"]) / f"{dataset}_answers"


def cmd_answer(cfg, args):
    """Генерация ответа по top-k пассажам и оценка судьёй для одного или нескольких run-/route-файлов."""
    import random

    from .data import load_corpus
    from .generation import CRITERIA, generate, judge

    tasks, _ = _tasks(cfg, args)
    if args.sample and args.sample < len(tasks):  # одна и та же выборка для всех систем
        ids = set(random.Random(args.seed).sample(sorted(t.task_id for t in tasks), args.sample))
        tasks = [t for t in tasks if t.task_id in ids]
    by = {t.task_id: t for t in tasks}
    llm, c = _llm(cfg), cfg["llm"]
    if c["generator_model"] == c["judge_model"]:
        print(f"ВНИМАНИЕ: генератор и судья — одна модель ({c['judge_model']}), оценки будут завышены")
    corpora = {d: load_corpus(cfg["data_dir"], d) for d in {t.domain for t in tasks}}

    def one(r):
        t = by[r["task_id"]]
        corpus = corpora[t.domain]
        docs = [d for d, _ in r["hits"][: args.k]]
        answer, g = generate(llm, c["generator_model"], t, docs, corpus, c["max_history_turns"])
        scores, j = judge(llm, c["judge_model"], t, answer, docs, corpus, c["max_history_turns"])
        return {"task_id": t.task_id, "answer": answer, "judge": scores, "docs": docs,
                "gen_tokens": g.prompt_tokens + g.completion_tokens, "gen_latency_ms": g.latency_ms,
                "judge_tokens": j.prompt_tokens + j.completion_tokens,
                "answerability": t.meta["answerability"]}

    for run in args.run:
        src = Path(run)
        recs = [r for r in read_jsonl(src) if r["task_id"] in by]
        if len(recs) < len(by):
            print(f"{src.name}: в файле {len(recs)} из {len(by)} задач выборки")
        name = ("route_" if src.parent.name.endswith("_routes") else "") + src.stem
        # одинаковые (вопрос, top-k пассажей) у разных систем берутся из кэша LLM
        out = parallel_map(one, recs, _workers(cfg, args), f"answer+judge {name}")
        dst = _answers_dir(cfg, args.dataset) / f"{name}.jsonl"
        write_jsonl(dst, out)
        ok = [o for o in out if o["judge"]]
        line = "  ".join(f"{crit}={sum(o['judge'][crit] for o in ok) / max(len(ok), 1):.3f}" for crit in CRITERIA)
        print(f"{name}: {line}  (оценено {len(ok)} из {len(out)}) → {dst}")


def cmd_answers_summary(cfg, args):
    """Таблица качества ответов по всем системам из runs/<dataset>_answers/ (+ парные тесты против baseline)."""
    import numpy as np

    from .generation import CRITERIA
    from .judging import answers_table
    from .stats import holm, paired_bootstrap, randomization_test

    tasks, _ = _tasks(cfg, args)
    slices, by = _slices(tasks)
    keep = lambda tid: tid in by and slices[args.slice](tid)  # noqa: E731
    files = sorted(_answers_dir(cfg, args.dataset).glob("*.jsonl"))
    rows, per_task = answers_table(files, keep)
    if not rows:
        raise SystemExit("нет файлов в runs/<dataset>_answers — сначала `arag answer --run ...`")
    print(f"slice={args.slice}  (шкала судьи 1–5; declined — доля ответов-отказов)\n")
    cols = ["system", "n", "judged", *CRITERIA, "declined", "gen_tokens", "judge_tokens"]
    _print_table(sorted(rows, key=lambda r: -(r["correctness"] or 0)), cols)
    out = Path(cfg["runs_dir"]) / f"{args.dataset}_answers_summary_{args.slice}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {out}")

    if args.baseline:
        if args.baseline not in per_task:
            raise SystemExit(f"нет ответов для {args.baseline}; есть: {', '.join(per_task)}")
        base, cmp_rows = per_task[args.baseline], []
        for s, sc in per_task.items():
            common = sorted(set(base) & set(sc))
            if s == args.baseline or not common:
                continue
            a = np.array([base[q][args.metric] for q in common], float)
            b = np.array([sc[q][args.metric] for q in common], float)
            groups = [by[q].conversation_id for q in common]
            delta, lo, hi = paired_bootstrap(a, b, args.n, groups=groups)
            cmp_rows.append({"system": s, "n": len(common), "baseline": a.mean(), "system_score": b.mean(),
                             "delta": delta, "ci_low": lo, "ci_high": hi, "p": randomization_test(a, b, args.n, groups=groups)})
        for r, p in zip(cmp_rows, holm([r["p"] for r in cmp_rows])):
            r["p_holm"] = p
        print(f"\nbaseline={args.baseline}  metric={args.metric}\n")
        _print_table(sorted(cmp_rows, key=lambda r: -r["delta"]),
                     ["system", "n", "baseline", "system_score", "delta", "ci_low", "ci_high", "p", "p_holm"])


def cmd_judge_calibrate(cfg, args):
    """Судья оценивает ответы с human-оценками из MTRAG; считаем согласие с людьми."""
    from .generation import judge_texts
    from .judging import calibration_report, load_human_eval

    items = load_human_eval(cfg["data_dir"])
    if args.limit:
        items = items[: args.limit]
    llm, c = _llm(cfg), cfg["llm"]
    model = args.model or c["judge_model"]

    def one(i):
        scores, _ = judge_texts(llm, model, i["turns"], i["reference"], i["answer"], i["passages"],
                                c["max_history_turns"])
        return i | {"judge": scores}

    items = parallel_map(one, items, args.workers or c.get("workers", 8), f"judge-calibrate {model}")
    report = calibration_report(items, model)
    out = Path(args.out) if args.out else Path(cfg["runs_dir"]) / f"judge_calibration_{model.replace('/', '__')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    write_jsonl(out.with_suffix(".jsonl"), [{k: i[k] for k in ("task_id", "model_id", "judge", "human", "official")}
                                            for i in items])
    print(report)
    print(f"\nsaved {out}")


def cmd_index(cfg, args):
    """Построить dense-индексы корпусов заранее (через API — с докачкой после обрыва)."""
    from .data import load_corpus

    engine = Engine(cfg, _llm(cfg))
    c = cfg["dense"]
    for d in args.domains:
        corpus = load_corpus(cfg["data_dir"], d)
        t0 = time.perf_counter()
        r = engine.dense(d)
        print(f"{d}: {len(corpus)} пассажей, матрица {r.emb.shape}, "
              f"{(time.perf_counter() - t0) / 60:.1f} мин (backend={c.get('backend', 'local')})")
        engine._dense.pop(d)  # не держим все матрицы в памяти


def cmd_rerank_check(cfg, args):
    """Загрузить reranker, проверить адекватность и замерить скорость на реальных пассажах (без кэша)."""
    from .data import load_corpus
    from .retrievers import Reranker

    c = cfg["reranker"]
    model = args.model or c["model"]
    t0 = time.perf_counter()
    rr = Reranker(model, args.batch_size or c.get("batch_size", 32), args.device or c.get("device"),
                  c.get("max_length", 512), cache_path=None)
    print(f"model: {model}  device: {rr.device}  загрузка {time.perf_counter() - t0:.1f} с")

    toy = {"nav": "The net asset value (NAV) of a mutual fund is its assets minus liabilities divided by shares.",
           "cap": "Market capitalization equals share price times the number of shares outstanding.",
           "k8s": "Kubernetes pods are the smallest deployable units of computing in a cluster."}
    s = rr.score("How is the NAV of a mutual fund calculated?", list(toy), toy)
    ok = s[0] > s[1] > s[2] or (s[0] > s[1] and s[0] > s[2])
    print(f"sanity: nav={s[0]:.3f} cap={s[1]:.3f} k8s={s[2]:.3f} → {'OK' if ok else 'ПОДОЗРИТЕЛЬНО'}")

    # скорость: реальные запросы и BM25-кандидаты (BM25 локальный и быстрый, к API не обращаемся)
    tasks, _ = load_tasks(cfg["data_dir"], "mtrag", [args.domain])
    tasks = [t for t in tasks if t.meta["has_qrels"]][: args.queries]
    engine = Engine(cfg, None)
    corpus = load_corpus(cfg["data_dir"], args.domain)
    n_cand = c["candidates"]
    jobs = [(t.query, [d for d, _ in engine.bm25(args.domain).search(t.query, n_cand)]) for t in tasks]
    rr.score(jobs[0][0], jobs[0][1][:4], corpus)  # прогрев
    t0 = time.perf_counter()
    pairs = 0
    for q, docs in jobs:
        rr.score(q, docs, corpus)
        pairs += len(docs)
    sec = time.perf_counter() - t0
    pps = pairs / sec
    per_query_ms = sec * 1000 / len(jobs)
    print(f"скорость ({args.domain}, {len(jobs)} запросов × {n_cand} кандидатов): {pps:.0f} пар/с, "
          f"{per_query_ms:.0f} мс на запрос")
    n = 777
    for name, k in [("1 стратегия", 1), ("4 ключевые (last/llm_rewrite × dense/hybrid)", 4), ("все 12 +rerank", 12)]:
        print(f"  {name}: ≤ {k * n * n_cand / pps / 60:.0f} мин (верхняя оценка; кэш пар уменьшает)")


def cmd_llm_check(cfg, args):
    """Без кэша: список моделей, тестовые rewrite и JSON-ответ роутера для каждой модели."""
    from .data import Task
    from .queries import QueryBuilder
    from .routers import LLMRouter

    c = cfg["llm"]
    key = c["api_key"]
    print(f"base_url: {c['base_url']}")
    print(f"api_key:  {'не задан (EMPTY)' if key == 'EMPTY' else f'задан, {len(key)} символов'}")
    print(f"extra_body: {c.get('extra_body') or '{}'}\n")
    llm = _llm(cfg)
    try:
        available = llm.list_models()
        print(f"/v1/models: {len(available)} моделей\n  " + "\n  ".join(available) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"/v1/models: ОШИБКА {e!r}\nПроверьте LLM_BASE_URL, LLM_API_KEY и доступ к сети (VPN).")
        return

    roles = {r: c[f"{r}_model"] for r in ("rewrite", "router", "generator", "judge")}
    models = args.models or list(dict.fromkeys(roles.values()))
    task = Task("check<::>2", "fiqa", [
        {"speaker": "user", "text": "What's the difference between Market Cap and NAV?"},
        {"speaker": "agent", "text": "Market cap is share price times shares outstanding; NAV is assets minus "
                                     "liabilities per share of a fund."},
        {"speaker": "user", "text": "Which one is more important?"}])

    ok_all = True
    for m in models:
        used_by = [r for r, mm in roles.items() if mm == m]
        print(f"=== {m}  (роли: {', '.join(used_by) or '—'})")
        if m not in available:
            print("  нет в /v1/models — проверьте имя в config.yaml")
            ok_all = False
            continue
        try:
            r = llm.chat(m, [{"role": "user", "content": "Reply with exactly one word: OK"}], max_tokens=64,
                         use_cache=False)
            think = "<think>" in r.text or "</think>" in r.text
            print(f"  ping: {r.text.strip()[:60]!r}  {r.latency_ms:.0f} ms, tokens {r.prompt_tokens}+"
                  f"{r.completion_tokens}, finish={r.finish_reason}{'  [в ответе есть thinking]' if think else ''}")
            if r.finish_reason == "length" or think:
                print("  ! модель рассуждает (thinking): ответы будут медленными и могут обрезаться по max_tokens.")

            class NoCache:  # тестовые вызовы мимо кэша
                def chat(self, model, messages, max_tokens=512, **kw):
                    return llm.chat(model, messages, max_tokens, use_cache=False, **kw)

            q = QueryBuilder(NoCache(), m).build(task, "llm_rewrite")
            print(f"  rewrite: {q.text!r}  {q.latency_ms:.0f} ms, completion {q.completion_tokens} tok")
            route = LLMRouter(NoCache(), m, use_history=True).route(task)
            parsed = route.rationale != "parse_error"
            print(f"  router JSON: {'OK' if parsed else 'НЕ РАЗОБРАН'} → {route.strategy}  ({route.rationale[:60]})")
            ok_all &= parsed
        except Exception as e:  # noqa: BLE001
            ok_all = False
            msg = repr(e)
            print(f"  ОШИБКА: {msg[:400]}")
            if c.get("extra_body") and ("chat_template_kwargs" in msg or "extra" in msg.lower()):
                print("  ! сервер не принимает extra_body — поставьте в config.yaml `extra_body: {}`")
    dc = cfg["dense"]
    if dc.get("backend") == "api":
        m = dc["model"]
        print(f"=== {m}  (dense, /v1/embeddings)")
        if m not in available:
            print("  нет в /v1/models — проверьте dense.model в config.yaml")
            ok_all = False
        else:
            try:
                import numpy as np

                v, ms = llm.embed_one(m, "Which one is more important, Market Cap or NAV?", use_cache=False)
                print(f"  query: dim={len(v)}, norm={np.linalg.norm(v):.3f}, {ms:.0f} ms")
                docs = ["Market capitalization is the share price multiplied by shares outstanding.",
                        "The net asset value (NAV) of a fund is its assets minus liabilities per share.",
                        "Kubernetes pods are the smallest deployable units."]
                e = llm.embed_batch(m, docs)
                sims = (e / np.linalg.norm(e, axis=1, keepdims=True)) @ (v / np.linalg.norm(v))
                print(f"  cosine с запросом: {np.round(sims, 3).tolist()} (последний текст не по теме — должен быть ниже)")
                long = ("word " * 2000)[: dc.get("max_chars", 8000)]
                t0 = time.perf_counter()
                llm.embed_batch(m, [long] * dc.get("batch_size", 64))
                print(f"  батч {dc.get('batch_size', 64)} × {len(long)} символов (max_chars): OK, "
                      f"{(time.perf_counter() - t0) * 1000:.0f} ms")
            except Exception as e:  # noqa: BLE001
                ok_all = False
                print(f"  ОШИБКА: {repr(e)[:400]}")
                print("  ! если ошибка про длину/размер — уменьшите dense.max_chars или dense.batch_size")
    print("\nвсё готово к прогонам" if ok_all else "\nесть проблемы, см. выше")


def main():
    p = argparse.ArgumentParser("arag")
    p.add_argument("--config")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--dataset", default="mtrag", choices=["mtrag", "mtragun"])
        sp.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
        sp.add_argument("--limit", type=int)
        sp.add_argument("--all-tasks", action="store_true", help="включить задачи без qrels")
        sp.add_argument("--workers", type=int, help="параллельных запросов к LLM (по умолчанию llm.workers)")

    sp = sub.add_parser("index", help="построить dense-индексы корпусов (через API — с докачкой)")
    sp.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)

    sp = sub.add_parser("rerank-check", help="проверить локальный reranker и замерить скорость")
    sp.add_argument("--model", help="модель вместо reranker.model из config.yaml")
    sp.add_argument("--domain", default="fiqa", choices=DOMAINS)
    sp.add_argument("--queries", type=int, default=20)
    sp.add_argument("--device", choices=["cpu", "mps", "cuda"], help="вместо reranker.device")
    sp.add_argument("--batch-size", type=int, help="вместо reranker.batch_size")

    sp = sub.add_parser("llm-check", help="проверить подключение к LLM API и модели из config.yaml")
    sp.add_argument("--models", nargs="+", help="проверить эти модели вместо моделей из конфига")

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
    sp.add_argument("--eps", type=float, default=0.05, help="допуск для cost-aware оракула")

    sp = sub.add_parser("analyze", help="анализ датасета, срезов и оракула → markdown")
    common(sp)
    sp.add_argument("--metric", default="ndcg_cut_5")
    sp.add_argument("--out", help="путь к .md (по умолчанию runs/<dataset>_analysis.md)")

    sp = sub.add_parser("significance",help="парные тесты: все runs и роутеры против baseline")
    common(sp)
    sp.add_argument("--baseline", required=True, help="стратегия (last:bm25) или route/<имя>")
    sp.add_argument("--metric", default="ndcg_cut_5")
    sp.add_argument("--slice", default="all", choices=["all", "first_turn", "later_turns"])
    sp.add_argument("--n", type=int, default=10000, help="число bootstrap/перестановок")

    sp = sub.add_parser("route", help="оценить роутер")
    common(sp)
    sp.add_argument("--router", default="rule",
                    choices=["fixed", "rule", "llm_nohist", "llm_hist", "llm_nohist_rewrite", "adaptive_rag",
                             "oracle", "oracle_cheap"])
    sp.add_argument("--fixed", default="last:hybrid+rerank")
    sp.add_argument("--train-dataset", choices=["mtrag", "mtragun"],
                    help="отдельный датасет обучения для adaptive_rag; без флага используется кросс-валидация")

    sp = sub.add_parser("routes-summary", help="сводная таблица роутеров: качество, решения, стоимость")
    common(sp)
    sp.add_argument("--slice", default="all", choices=["all", "first_turn", "later_turns"])

    sp = sub.add_parser("answer", help="генерация ответа и LLM-as-a-Judge по run-/route-файлам")
    common(sp)
    sp.add_argument("--run", required=True, nargs="+",
                    help="пути к runs/<dataset>/<strategy>.jsonl и/или runs/<dataset>_routes/<router>.jsonl")
    sp.add_argument("--k", type=int, default=5, help="сколько пассажей давать генератору")
    sp.add_argument("--sample", type=int, help="случайная подвыборка задач (одна и та же для всех систем)")
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("answers-summary", help="таблица качества ответов по системам")
    common(sp)
    sp.add_argument("--slice", default="all", choices=["all", "first_turn", "later_turns"])
    sp.add_argument("--baseline", help="система для парных тестов: стратегия или route/<имя>")
    sp.add_argument("--metric", default="correctness", choices=["correctness", "faithfulness", "completeness"])
    sp.add_argument("--n", type=int, default=10000)

    sp = sub.add_parser("judge-calibrate", help="согласие LLM-судьи с human-оценками MTRAG")
    sp.add_argument("--model", help="модель судьи вместо llm.judge_model")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--workers", type=int)
    sp.add_argument("--out", help="путь к .md (по умолчанию runs/judge_calibration_<model>.md)")

    args = p.parse_args()
    cfg = load_config(args.config)
    cmds = {"run": cmd_run, "eval": cmd_eval, "oracle": cmd_oracle, "analyze": cmd_analyze,
            "significance": cmd_significance, "route": cmd_route, "answer": cmd_answer, "llm-check": cmd_llm_check,
            "index": cmd_index, "rerank-check": cmd_rerank_check,
            "routes-summary": cmd_routes_summary, "answers-summary": cmd_answers_summary,
            "judge-calibrate": cmd_judge_calibrate}
    cmds[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
