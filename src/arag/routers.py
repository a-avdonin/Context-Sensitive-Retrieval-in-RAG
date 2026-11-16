"""Роутеры выбирают стратегию retrieval для задачи.

fixed        — один пайплайн для всех запросов
rule         — эвристики по запросу и истории: переписывать только зависимые от контекста вопросы
llm_nohist   — LLM видит только последний вопрос (аналог Adaptive-RAG: адаптация по сложности)
llm_hist     — LLM видит вопрос и историю, решает про rewrite, ретривер и reranking
adaptive_rag — Adaptive-RAG: классификатор сложности по одному вопросу выбирает уровень A/B/C
oracle       — лучшая стратегия по разметке оракула (верхняя граница, не реализуем)
oracle_cheap — самая дешёвая стратегия в пределах eps от лучшей (верхняя граница с учётом стоимости)
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .data import Task
from .llm import LLM, format_history, parse_json
from .pipeline import Strategy


@dataclass
class Route:
    strategy: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    rationale: str = ""


class FixedRouter:
    def __init__(self, strategy: str):
        self.strategy = strategy

    def route(self, task: Task) -> Route:
        return Route(self.strategy)


ANAPHORA = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|there|the same|above|former|latter|"
    r"what about|how about|and if|why not|which one|more|else|also)\b",
    re.I,
)


class RuleRouter:
    """Переписываем запрос, только если он зависит от контекста (местоимение/анафора, очень короткий или
    без вопросительной формы). Стратегии для трёх случаев задаются в config.yaml (routers.rule)."""

    def __init__(self, first_turn: str = "last:dense+rerank", standalone: str = "last:dense+rerank",
                 dependent: str = "llm_rewrite:dense+rerank", short_len: int = 3):
        for s in (first_turn, standalone, dependent):
            Strategy.parse(s)
        self.first_turn, self.standalone, self.dependent = first_turn, standalone, dependent
        self.short_len = short_len

    def features(self, task: Task) -> dict:
        q = task.query
        return {
            "first_turn": not task.history,
            "anaphora": bool(ANAPHORA.search(q)),
            "short": len(q.split()) <= self.short_len,
            "no_question": not re.search(r"\b(what|how|why|when|where|who|which|can|is|are|do|does)\b|\?", q, re.I),
        }

    def route(self, task: Task) -> Route:
        f = self.features(task)
        if f["first_turn"]:
            return Route(self.first_turn, rationale="first turn")
        if not (f["anaphora"] or f["short"] or f["no_question"]):
            return Route(self.standalone, rationale="standalone")
        return Route(self.dependent, rationale=",".join(k for k, v in f.items() if v))


class OracleRouter:
    """Читает разметку `arag oracle` (runs/<dataset>_oracle.jsonl). Это верхняя граница, а не реализуемый роутер."""

    def __init__(self, path: Path, key: str = "best", fallback: str = "last:dense"):
        if not path.exists():
            raise FileNotFoundError(f"{path}: сначала выполните `arag oracle`")
        self.labels = {}
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                self.labels[r["task_id"]] = r[key]
        self.key, self.fallback = key, fallback

    def route(self, task: Task) -> Route:
        s = self.labels.get(task.task_id)
        return Route(s or self.fallback, rationale=self.key if s else "no_label")


PROMPT_HIST = """You configure a retrieval pipeline for a conversational QA system.
Given the conversation and the user's last question, choose:
- "rewrite": true if the last question depends on earlier turns (pronouns, ellipsis, topic shift that needs context) and must be rewritten into a standalone query; false if it is already self-contained.
- "retriever": "bm25" for keyword-heavy queries (names, codes, exact terms), "dense" for paraphrased or conceptual queries, "hybrid" when unsure or both apply.
- "rerank": true if the question is complex, ambiguous or needs precise passages; false for simple lookups.

Conversation:
{history}

Last question: {query}

Answer with JSON only: {{"rewrite": bool, "retriever": "bm25|dense|hybrid", "rerank": bool, "reason": "short"}}"""

PROMPT_NOHIST = """You configure a retrieval pipeline for a QA system.
Given the user's question, choose:
- "retriever": "bm25" for keyword-heavy queries (names, codes, exact terms), "dense" for paraphrased or conceptual queries, "hybrid" when unsure or both apply.
- "rerank": true if the question is complex, ambiguous or needs precise passages; false for simple lookups.

Question: {query}

Answer with JSON only: {{"retriever": "bm25|dense|hybrid", "rerank": bool, "reason": "short"}}"""


class LLMRouter:
    def __init__(self, llm: LLM, model: str, use_history: bool, max_history_turns: int = 6,
                 fallback: str = "last:dense"):
        self.llm, self.model, self.use_history = llm, model, use_history
        self.max_turns, self.fallback = max_history_turns, fallback

    def route(self, task: Task) -> Route:
        if self.use_history:
            prompt = PROMPT_HIST.format(history=format_history(task.history, self.max_turns) or "(empty)",
                                        query=task.query)
        else:
            prompt = PROMPT_NOHIST.format(query=task.query)
        r = self.llm.chat(self.model, [{"role": "user", "content": prompt}], max_tokens=200)
        try:
            d = parse_json(r.text)
            query = "llm_rewrite" if self.use_history and d.get("rewrite") else "last"
            strategy = str(Strategy(query, d["retriever"], bool(d.get("rerank"))))
            Strategy.parse(strategy)
        except Exception:
            strategy, d = self.fallback, {"reason": "parse_error"}
        return Route(strategy, r.prompt_tokens, r.completion_tokens, r.latency_ms, d.get("reason", ""))


class AdaptiveRAGRouter:
    """Adaptive-RAG (Jeong et al., NAACL 2024), адаптация к retrieval в диалоге.

    Как в оригинале: классификатор сложности видит ТОЛЬКО текущий вопрос (без истории) и выбирает один из
    уровней пайплайна возрастающей стоимости; метки строятся автоматически — самый простой уровень, который
    даёт лучший результат на запросе (у авторов — самый простой, ответивший правильно).
    Отличия: уровни — наши стратегии (A/B/C из config), а не no-retrieval/single-step/multi-step;
    классификатор — логистическая регрессия на эмбеддингах bge-m3, а не дообученный T5-large;
    запросы, где все уровни дают 0, в обучение не идут (у авторов им ставится метка по типу датасета).
    Предсказания — кросс-валидация по диалогам (GroupKFold): модель не видела диалог задачи.
    """

    def __init__(self, llm, emb_model: str, query_prefix: str, levels: list[str], runs_dir: Path, dataset: str,
                 metric: str = "ndcg_cut_5", eps: float = 0.0, folds: int = 5, C: float = 1.0, workers: int = 8):
        for s in levels:
            Strategy.parse(s)
        self.llm, self.emb_model, self.prefix = llm, emb_model, query_prefix
        self.levels, self.runs_dir, self.dataset = levels, Path(runs_dir), dataset
        self.metric, self.eps, self.folds, self.C, self.workers = metric, eps, folds, C, workers
        self.pred: dict[str, tuple[int, float]] = {}
        self.embed_ms: dict[str, float] = {}
        self.report: dict = {}

    def _level_scores(self, tasks, qrels) -> dict[str, list[float]]:
        from .metrics import per_query, read_jsonl, records_to_run

        ids = {t.task_id for t in tasks}
        out = {}
        for s in self.levels:
            path = self.runs_dir / self.dataset / f"{s.replace(':', '__')}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"{path}: нужен run уровня {s} (arag run --strategies {s})")
            sc = per_query(qrels, records_to_run([r for r in read_jsonl(path) if r["task_id"] in ids]))
            for tid in ids:
                out.setdefault(tid, []).append(sc.get(tid, {}).get(self.metric, 0.0))
        return out

    @staticmethod
    def label(scores: list[float], eps: float = 0.0) -> int | None:
        """Самый простой уровень с результатом не хуже лучшего − eps; None, если все уровни дали 0."""
        best = max(scores)
        if best <= 0:
            return None
        return next(i for i, v in enumerate(scores) if v >= best - eps - 1e-12)

    def prepare(self, tasks, qrels):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold

        from .llm import parallel_map

        scores = self._level_scores(tasks, qrels)
        labels = [self.label(scores[t.task_id], self.eps) for t in tasks]
        embs = parallel_map(lambda t: self.llm.embed_one(self.emb_model, self.prefix + t.query), tasks,
                            self.workers, "adaptive_rag: query embeddings")
        X = np.stack([v for v, _ in embs]).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self.embed_ms = {t.task_id: ms for t, (_, ms) in zip(tasks, embs)}
        groups = np.array([t.task_id.split("<::>")[0] for t in tasks])
        y = np.array([-1 if l is None else l for l in labels])
        pred = np.zeros(len(tasks), dtype=int)
        conf = np.zeros(len(tasks))
        for tr, te in GroupKFold(n_splits=self.folds).split(X, y, groups):
            tr = tr[y[tr] >= 0]
            if len(set(y[tr])) == 1:
                pred[te], conf[te] = y[tr][0], 1.0
                continue
            clf = LogisticRegression(C=self.C, max_iter=2000).fit(X[tr], y[tr])
            proba = clf.predict_proba(X[te])
            pred[te] = clf.classes_[proba.argmax(1)]
            conf[te] = proba.max(1)
        self.pred = {t.task_id: (int(p), float(c)) for t, p, c in zip(tasks, pred, conf)}

        lab = y >= 0
        k = len(self.levels)
        cm = [[int(((y == i) & (pred == j)).sum()) for j in range(k)] for i in range(k)]
        self.report = {
            "levels": self.levels,
            "label_dist": [int((y == i).sum()) for i in range(k)],
            "unlabeled": int((~lab).sum()),
            "pred_dist": [int((pred == i).sum()) for i in range(k)],
            "cv_accuracy": float((pred[lab] == y[lab]).mean()) if lab.any() else 0.0,
            "majority_accuracy": float(max(np.bincount(y[lab], minlength=k)) / lab.sum()) if lab.any() else 0.0,
            "confusion": cm,
        }
        return self.report

    def route(self, task: Task) -> Route:
        if task.task_id not in self.pred:
            raise RuntimeError("AdaptiveRAGRouter.prepare(tasks, qrels) не вызван для этой задачи")
        lvl, p = self.pred[task.task_id]
        # стоимость роутера: эмбеддинг вопроса (latency первого вызова) + классификатор (~0)
        return Route(self.levels[lvl], latency_ms=self.embed_ms.get(task.task_id, 0.0),
                     rationale=f"level={'ABCDEFG'[lvl]} p={p:.2f}")


def build_router(name: str, cfg: dict, llm: LLM | None = None, fixed: str = "last:hybrid+rerank",
                 dataset: str = "mtrag"):
    model, turns = cfg["llm"]["router_model"], cfg["llm"]["max_history_turns"]
    if name == "fixed":
        return FixedRouter(fixed)
    if name == "rule":
        return RuleRouter(**cfg.get("routers", {}).get("rule", {}))
    if name == "adaptive_rag":
        c, d = cfg.get("routers", {}).get("adaptive_rag", {}), cfg["dense"]
        return AdaptiveRAGRouter(llm, d["model"], d.get("query_prefix", ""),
                                 c.get("levels", ["last:dense", "last:dense+rerank", "llm_rewrite:dense+rerank"]),
                                 Path(cfg["runs_dir"]), dataset, cfg["eval"].get("oracle_metric", "ndcg_cut_5"),
                                 c.get("eps", 0.0), c.get("folds", 5), c.get("C", 1.0),
                                 cfg["llm"].get("workers", 8))
    if name in ("oracle", "oracle_cheap"):
        return OracleRouter(Path(cfg["runs_dir"]) / f"{dataset}_oracle.jsonl",
                            "best" if name == "oracle" else "best_cheap")
    if name == "llm_nohist":
        return LLMRouter(llm, model, use_history=False, max_history_turns=turns)
    if name == "llm_hist":
        return LLMRouter(llm, model, use_history=True, max_history_turns=turns)
    raise ValueError(name)
