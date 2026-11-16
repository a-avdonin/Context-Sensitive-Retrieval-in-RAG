"""Анализ датасета, качества стратегий по срезам и оракула. Результат — markdown-отчёт."""
import re
from collections import Counter, defaultdict

from .data import Task
from .routers import RuleRouter

try:
    from bm25s.stopwords import STOPWORDS_EN as _SW
except ImportError:  # pragma: no cover
    _SW = ()
STOP = set(_SW) | {"what", "how", "why", "when", "where", "who", "which", "whose", "whom", "does", "do", "did",
                     "can", "could", "would", "should", "is", "are", "was", "were", "about", "tell", "me"}
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def content_tokens(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 1}


def turn_bucket(t: Task) -> str:
    n = t.user_turn
    return str(n) if n <= 5 else "6+"


def rewrite_stats(t: Task) -> dict | None:
    """Чего не хватает последнему вопросу относительно human rewrite и можно ли это достать из истории."""
    if not t.history or not t.gold_rewrite:
        return None
    last, gold = content_tokens(t.query), content_tokens(t.gold_rewrite)
    hist = set().union(*(content_tokens(x["text"]) for x in t.history))
    missing = gold - last
    return {
        "identical": t.query.strip().lower().rstrip("?. ") == t.gold_rewrite.strip().lower().rstrip("?. "),
        "len_last": len(t.query.split()),
        "len_gold": len(t.gold_rewrite.split()),
        "jaccard": len(last & gold) / max(len(last | gold), 1),
        "deficit": len(missing) / max(len(gold), 1),
        "missing_in_history": len(missing & hist) / len(missing) if missing else None,
        "missing_in_user_history": (len(missing & set().union(*(content_tokens(x["text"]) for x in t.history
                                                                  if x["speaker"] == "user"))) / len(missing)
                                    if missing else None),
    }


def task_features(t: Task) -> dict:
    f = RuleRouter().features(t)
    return {
        "domain": t.domain,
        "turn": turn_bucket(t),
        "later": t.user_turn > 1,
        "multi_turn": t.meta.get("multi_turn") or "N/A",
        "answerability": t.meta.get("answerability") or "N/A",
        "question_type": (t.meta.get("question_type") or ["N/A"])[0],
        "n_relevant": None,
        "anaphora": f["anaphora"],
        "short": f["short"],
        "dependent": "first_turn" if f["first_turn"] else (f["anaphora"] or f["short"] or f["no_question"]),
    }


# ---------- markdown ----------

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def build_report(tasks: list[Task], qrels: dict, scores: dict[str, dict[str, dict]], metric: str,
                 dataset: str) -> str:
    """scores: стратегия → task_id → метрики pytrec_eval."""
    by = {t.task_id: t for t in tasks}
    feats = {t.task_id: task_features(t) | {"n_relevant": len(qrels.get(t.task_id, {}))} for t in tasks}
    strategies = sorted(scores, key=lambda s: -_mean([v[metric] for v in scores[s].values()]) or 0)
    S = []

    S.append(f"# Анализ: {dataset}\n\nЗадач с qrels: {len(tasks)}. Метрика: `{metric}`. "
             f"Стратегии: {', '.join(f'`{s}`' for s in strategies)}.\n")

    # 1. состав датасета
    S.append("## 1. Состав датасета\n")
    for key, title in [("domain", "Домен"), ("turn", "Номер хода пользователя"), ("multi_turn", "Multi-Turn"),
                       ("answerability", "Answerability"), ("question_type", "Тип вопроса")]:
        c = Counter(f[key] for f in feats.values())
        later = Counter(f[key] for f in feats.values() if f["later"])
        rows = [{title: k, "задач": v, "из них later": later.get(k, 0), "доля": v / len(feats)}
                for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0]))]
        S.append(f"### {title}\n\n" + md_table(rows, [title, "задач", "из них later", "доля"]) + "\n")
    nrel = [f["n_relevant"] for f in feats.values()]
    S.append(f"Релевантных пассажей на запрос: среднее {_mean(nrel):.2f}, "
             f"распределение {dict(sorted(Counter(min(n, 5) for n in nrel).items()))} (5 = 5+).\n")

    # 2. last vs gold rewrite
    rs = {tid: rewrite_stats(t) for tid, t in by.items()}
    later_ids = [tid for tid, f in feats.items() if f["later"]]
    have = [tid for tid in later_ids if rs[tid]]
    S.append("## 2. Последний вопрос vs human rewrite (later turns)\n")
    S.append(f"Human rewrite есть у {len(have)} из {len(later_ids)} later-задач; у остальных "
             f"`gold_rewrite` = последний вопрос.\n")
    if have:
        g = [rs[t] for t in have]
        rows = [
            {"показатель": "rewrite совпадает с вопросом", "значение": _mean([float(x["identical"]) for x in g])},
            {"показатель": "длина вопроса, слов", "значение": _mean([x["len_last"] for x in g])},
            {"показатель": "длина rewrite, слов", "значение": _mean([x["len_gold"] for x in g])},
            {"показатель": "Jaccard контентных токенов", "значение": _mean([x["jaccard"] for x in g])},
            {"показатель": "дефицит: доля токенов rewrite, которых нет в вопросе", "значение": _mean([x["deficit"] for x in g])},
            {"показатель": "из недостающих токенов есть в истории (все реплики)", "значение": _mean([x["missing_in_history"] for x in g])},
            {"показатель": "из недостающих токенов есть в репликах пользователя", "значение": _mean([x["missing_in_user_history"] for x in g])},
        ]
        S.append(md_table(rows, ["показатель", "значение"]) + "\n")

    # 3. качество по срезам
    S.append(f"## 3. Качество стратегий по срезам (`{metric}`)\n")
    for key in ["turn", "domain", "multi_turn", "answerability", "question_type", "dependent"]:
        groups = defaultdict(list)
        for tid, f in feats.items():
            groups[f[key]].append(tid)
        rows = []
        for gname, ids in sorted(groups.items(), key=lambda x: str(x[0])):
            r = {key: gname, "n": len(ids)}
            for s in strategies:
                r[s] = _mean([scores[s][i][metric] for i in ids if i in scores[s]])
            rows.append(r)
        note = " (`dependent` — эвристика RuleRouter: later-ход с местоимением, короткий или без вопроса; first_turn — первый ход)" \
            if key == "dependent" else ""
        S.append(f"### По `{key}`{note}\n\n" + md_table(rows, [key, "n"] + strategies) + "\n")

    # 4. оракул
    S.append("## 4. Оракул\n")
    deploy = [s for s in strategies if not s.startswith("gold_rewrite")]
    best_dep = {tid: max((scores[s][tid][metric] for s in deploy if tid in scores[s]), default=0.0) for tid in feats}
    best_all = {tid: max((scores[s][tid][metric] for s in strategies if tid in scores[s]), default=0.0) for tid in feats}
    rows = []
    for key in ["domain", "turn"]:
        groups = defaultdict(list)
        for tid, f in feats.items():
            groups[f[key]].append(tid)
        for gname, ids in sorted(groups.items(), key=lambda x: str(x[0])):
            rows.append({"срез": f"{key}={gname}", "n": len(ids),
                         "unsolvable (deployable)": _mean([float(best_dep[i] == 0) for i in ids]),
                         "unsolvable (+gold)": _mean([float(best_all[i] == 0) for i in ids]),
                         "oracle deployable": _mean([best_dep[i] for i in ids]),
                         "oracle +gold": _mean([best_all[i] for i in ids])})
    S.append("Unsolvable — доля запросов, где ни одна стратегия не дала ненулевой метрики.\n\n" +
             md_table(rows, ["срез", "n", "unsolvable (deployable)", "unsolvable (+gold)",
                             "oracle deployable", "oracle +gold"]) + "\n")

    # 5. когда rewrite помогает
    if "gold_rewrite:bm25" in scores and "last:bm25" in scores:
        gs, ls = scores["gold_rewrite:bm25"], scores["last:bm25"]
        S.append("## 5. Когда human rewrite помогает BM25 (later turns с rewrite)\n")
        S.append("win/tie/loss — сравнение `gold_rewrite:bm25` с `last:bm25` по запросу.\n")
        rows = []
        for key in ["multi_turn", "dependent", "question_type"]:
            groups = defaultdict(list)
            for tid in have:
                groups[feats[tid][key]].append(tid)
            for gname, ids in sorted(groups.items(), key=lambda x: str(x[0])):
                d = [gs[i][metric] - ls[i][metric] for i in ids if i in gs and i in ls]
                if not d:
                    continue
                rows.append({"срез": f"{key}={gname}", "n": len(d), "Δ mean": _mean(d),
                             "win": sum(x > 0 for x in d), "tie": sum(x == 0 for x in d),
                             "loss": sum(x < 0 for x in d)})
        S.append(md_table(rows, ["срез", "n", "Δ mean", "win", "tie", "loss"]) + "\n")

        # дефицит контекста vs выигрыш
        bins = [(0, 0.001, "0"), (0.001, 0.34, "(0, 1/3)"), (0.34, 0.67, "[1/3, 2/3)"), (0.67, 1.01, "[2/3, 1]")]
        rows = []
        for lo, hi, name in bins:
            ids = [t for t in have if lo <= rs[t]["deficit"] < hi and t in gs and t in ls]
            if ids:
                d = [gs[i][metric] - ls[i][metric] for i in ids]
                rows.append({"дефицит": name, "n": len(ids), "last": _mean([ls[i][metric] for i in ids]),
                             "gold": _mean([gs[i][metric] for i in ids]), "Δ mean": _mean(d),
                             "win": sum(x > 0 for x in d), "loss": sum(x < 0 for x in d)})
        S.append("### Выигрыш от rewrite в зависимости от дефицита контекста\n\n" +
                 md_table(rows, ["дефицит", "n", "last", "gold", "Δ mean", "win", "loss"]) + "\n")
    return "\n".join(S)
