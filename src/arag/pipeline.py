"""Стратегия retrieval = <query_mode>:<retriever>[+rerank], например llm_rewrite:hybrid+rerank."""
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Task, load_corpus
from .queries import QUERY_MODES, QueryBuilder
from .retrievers import APIDenseRetriever, BM25Retriever, DenseRetriever, Reranker, rrf

RETRIEVERS = ["bm25", "dense", "hybrid"]


@dataclass(frozen=True)
class Strategy:
    query: str
    retriever: str
    rerank: bool = False

    @classmethod
    def parse(cls, s: str) -> "Strategy":
        query, rest = s.split(":")
        retriever, _, rr = rest.partition("+")
        assert query in QUERY_MODES and retriever in RETRIEVERS and rr in ("", "rerank"), s
        return cls(query, retriever, rr == "rerank")

    def __str__(self):
        return f"{self.query}:{self.retriever}{'+rerank' if self.rerank else ''}"


def all_strategies() -> list[str]:
    return [str(Strategy(q, r, rr)) for q in QUERY_MODES for r in RETRIEVERS for rr in (False, True)]


@dataclass
class RunRecord:
    task_id: str
    strategy: str
    query: str
    hits: list[tuple[str, float]]
    latency_ms: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rerank_pairs: int = 0
    reranker: str = ""  # модель reranker'а: при смене модели run надо пересчитать (--overwrite)
    query_info: dict | None = None  # промежуточные шаги формирования запроса (CHIQ)


class Engine:
    """Лениво поднимает индексы и модели на домен; выполняет стратегию на задаче."""

    def __init__(self, cfg: dict, llm=None):
        self.cfg, self.llm = cfg, llm
        self.qb = QueryBuilder(llm, cfg["llm"]["rewrite_model"], cfg["llm"]["max_history_turns"])
        self._bm25, self._dense, self._reranker = {}, {}, None

    def corpus(self, domain: str) -> dict[str, str]:
        return load_corpus(self.cfg["data_dir"], domain)

    def _cache(self, domain: str) -> Path:
        p = Path(self.cfg["cache_dir"]) / "index" / domain
        p.mkdir(parents=True, exist_ok=True)
        return p

    def bm25(self, domain):
        if domain not in self._bm25:
            self._bm25[domain] = BM25Retriever(self.corpus(domain), self._cache(domain))
        return self._bm25[domain]

    def dense(self, domain):
        if domain not in self._dense:
            c = self.cfg["dense"]
            if c.get("backend", "local") == "api":
                if self.llm is None:
                    raise RuntimeError("dense.backend=api требует LLM-клиент (LLM_BASE_URL / LLM_API_KEY)")
                self._dense[domain] = APIDenseRetriever(self.llm, c["model"], self.corpus(domain), self._cache(domain),
                                                        c.get("query_prefix", ""), c.get("batch_size", 64),
                                                        c.get("workers", 4), c.get("max_chars", 8000), desc=domain)
            else:
                self._dense[domain] = DenseRetriever(self.corpus(domain), self._cache(domain), c["model"],
                                                     c["query_prefix"], c["batch_size"], c["device"])
        return self._dense[domain]

    def query_text(self, task: Task, s: Strategy | str) -> str:
        s = Strategy.parse(s) if isinstance(s, str) else s
        return self.qb.build(task, s.query).text

    def reranker(self):
        if self._reranker is None:
            c = self.cfg["reranker"]
            self._reranker = Reranker(c["model"], c.get("batch_size", 32), c.get("device"), c.get("max_length", 512),
                                      Path(self.cfg["cache_dir"]) / "rerank_cache.sqlite")
        return self._reranker

    def prepare(self, domain: str, s: Strategy):
        """Загрузка индексов и моделей вне замера latency."""
        if s.retriever in ("bm25", "hybrid"):
            self.bm25(domain)
        if s.retriever in ("dense", "hybrid"):
            self.dense(domain)
        if s.rerank:
            self.reranker()

    def run(self, task: Task, strategy: Strategy | str) -> RunRecord:
        s = Strategy.parse(strategy) if isinstance(strategy, str) else strategy
        rc = self.cfg["retrieval"]
        lat = {}

        self.prepare(task.domain, s)
        q = self.qb.build(task, s.query)
        lat["query"] = q.latency_ms

        t0 = time.perf_counter()
        n = rc["first_stage_k"] if s.rerank or s.retriever == "hybrid" else rc["top_k"]
        if s.retriever == "bm25":
            hits = self.bm25(task.domain).search(q.text, n)
        elif s.retriever == "dense":
            hits = self.dense(task.domain).search(q.text, n)
        else:
            hits = rrf([self.bm25(task.domain).search(q.text, n), self.dense(task.domain).search(q.text, n)],
                       n, rc["rrf_k"])
        wall = (time.perf_counter() - t0) * 1000
        if s.retriever in ("dense", "hybrid"):
            d = self.dense(task.domain)
            # эмбеддинг запроса — отдельный этап; при попадании в кэш берём latency первого вызова
            lat["embed"] = d.embed_ms
            wall -= d.embed_wall_ms
        lat["retrieve"] = max(wall, 0.0)

        pairs = 0
        if s.rerank:
            cand = hits[: self.cfg["reranker"]["candidates"]]
            pairs = len(cand)
            rr = self.reranker()
            hits = rr.rerank(q.rerank_text or q.text, cand, self.corpus(task.domain), rc["top_k"])
            lat["rerank"] = rr.last_latency_ms  # per-pair время первого расчёта; кэш не занижает стоимость
        hits = hits[: rc["top_k"]]
        lat["total"] = sum(lat.values())

        return RunRecord(task.task_id, str(s), q.text, hits, lat, q.prompt_tokens, q.completion_tokens, pairs,
                         self.cfg["reranker"]["model"] if s.rerank else "", q.info or None)


def record_to_dict(r: RunRecord) -> dict:
    return asdict(r)
