import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

Hits = list[tuple[str, float]]


class BM25Retriever:
    def __init__(self, corpus: dict[str, str], cache_dir: Path):
        import bm25s
        import Stemmer

        self.stemmer = Stemmer.Stemmer("english")
        self.ids = list(corpus)
        path = cache_dir / "bm25"
        if (path / "params.index.json").exists():
            self.model = bm25s.BM25.load(path)
        else:
            tokens = bm25s.tokenize(list(corpus.values()), stopwords="en", stemmer=self.stemmer, show_progress=False)
            self.model = bm25s.BM25()
            self.model.index(tokens, show_progress=False)
            self.model.save(path)

        self.tokenize = lambda q: bm25s.tokenize([q], stopwords="en", stemmer=self.stemmer, show_progress=False)

    def search(self, query: str, k: int) -> Hits:
        docs, scores = self.model.retrieve(self.tokenize(query), k=k, show_progress=False)
        return [(self.ids[i], float(s)) for i, s in zip(docs[0], scores[0])]


class _DenseBase:
    """Поиск по нормированным эмбеддингам (cosine = dot). Матрица в float32: fp16-matmul в numpy очень медленный.

    После search() выставляются embed_ms (latency эмбеддинга запроса для отчёта; при попадании в кэш —
    latency первого вызова) и embed_wall_ms (сколько реально заняло), чтобы Engine мог их разделить.
    """

    ids: list[str]
    emb: np.ndarray
    embed_ms: float = 0.0
    embed_wall_ms: float = 0.0

    def encode_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, k: int) -> Hits:
        q = self.encode_query(query)
        scores = self.emb @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class DenseRetriever(_DenseBase):
    """Локальная модель sentence-transformers (нужен extra [neural] и желательно GPU)."""

    def __init__(self, corpus: dict[str, str], cache_dir: Path, model_name: str, query_prefix: str = "",
                 batch_size: int = 128, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = query_prefix
        self.ids = list(corpus)
        path = cache_dir / f"dense_{model_name.replace('/', '__')}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            emb = self.model.encode(list(corpus.values()), batch_size=batch_size, normalize_embeddings=True,
                                    show_progress_bar=True, convert_to_numpy=True).astype(np.float16)
            np.save(path, emb)
        self.emb = emb.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        t0 = time.perf_counter()
        q = self.model.encode([self.prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        self.embed_ms = self.embed_wall_ms = (time.perf_counter() - t0) * 1000
        return q.astype(np.float32)


# ---------- эмбеддинги через OpenAI-совместимый API (/v1/embeddings) ----------

def api_index_path(cache_dir: Path, model: str) -> Path:
    return cache_dir / f"dense_api__{model.replace('/', '__')}.npy"


def _ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def build_api_index(llm, model: str, corpus: dict[str, str], cache_dir: Path, batch_size: int = 64,
                    workers: int = 4, max_chars: int = 8000, shard_batches: int = 16, desc: str = "") -> Path:
    """Закодировать корпус через API. Пишет шарды по batch_size*shard_batches пассажей: прерванный
    прогон при перезапуске досчитывает только недостающие шарды. В конце склеивает в один .npy (fp16)."""
    from .llm import parallel_map

    path = api_index_path(cache_dir, model)
    meta_path = path.with_suffix(".json")
    ids = list(corpus)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("ids") == _ids_digest(ids) and meta.get("n") == len(ids):
            return path
        raise RuntimeError(f"{path} не совпадает с корпусом; удалите его и пересоберите индекс")

    texts = [t[:max_chars] for t in corpus.values()]
    shard = batch_size * shard_batches
    shard_dir = path.with_suffix(".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(texts) + shard - 1) // shard
    todo = [i for i in range(n_shards) if not (shard_dir / f"{i:05d}.npy").exists()]

    def encode_shard(i: int):
        chunk = texts[i * shard:(i + 1) * shard]
        parts = [llm.embed_batch(model, chunk[j:j + batch_size]) for j in range(0, len(chunk), batch_size)]
        tmp = shard_dir / f"{i:05d}.tmp.npy"
        np.save(tmp, _normalize(np.concatenate(parts)).astype(np.float16))
        tmp.rename(shard_dir / f"{i:05d}.npy")

    if todo:
        print(f"{desc}: {len(texts)} пассажей, шардов {n_shards}, осталось {len(todo)}")
        parallel_map(encode_shard, todo, workers, f"index {desc}")
    emb = np.concatenate([np.load(shard_dir / f"{i:05d}.npy") for i in range(n_shards)])
    assert len(emb) == len(ids), (len(emb), len(ids))
    np.save(path, emb)
    meta_path.write_text(json.dumps({"model": model, "n": len(ids), "dim": int(emb.shape[1]),
                                     "ids": _ids_digest(ids), "max_chars": max_chars}))
    shutil.rmtree(shard_dir)
    return path


class APIDenseRetriever(_DenseBase):
    def __init__(self, llm, model: str, corpus: dict[str, str], cache_dir: Path, query_prefix: str = "",
                 batch_size: int = 64, workers: int = 4, max_chars: int = 8000, desc: str = ""):
        self.llm, self.model, self.prefix = llm, model, query_prefix
        self.ids = list(corpus)
        path = build_api_index(llm, model, corpus, cache_dir, batch_size, workers, max_chars, desc=desc)
        self.emb = np.load(path).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        t0 = time.perf_counter()
        vec, self.embed_ms = self.llm.embed_one(self.model, self.prefix + query)
        self.embed_wall_ms = (time.perf_counter() - t0) * 1000
        return _normalize(vec)


def rrf(runs: list[Hits], k: int, rrf_k: int = 60) -> Hits:
    fused: dict[str, float] = {}
    for hits in runs:
        for rank, (doc, _) in enumerate(hits):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


def auto_device(device: str | None = None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Reranker:
    """Cross-encoder (sentence-transformers) с кэшем оценок пар (запрос, пассаж) на sqlite.

    Разные стратегии часто дают одни и те же кандидаты для одного текста запроса, поэтому пары
    считаются один раз. Latency для отчёта = сумма per-pair времени, измеренного при первом расчёте
    (как будто пары считались заново), чтобы кэш не занижал стоимость стратегии.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None, max_length: int = 512,
                 cache_path: Path | None = None, model=None):
        import sqlite3
        import threading

        self.model_name, self.batch_size = model_name, batch_size
        if model is None:
            from sentence_transformers import CrossEncoder

            self.device = auto_device(device)
            model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        else:
            self.device = "fake"
        self.model = model
        self.last_latency_ms = 0.0
        self.last_computed = 0
        self.db = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._lock = threading.Lock()
            self.db = sqlite3.connect(cache_path, check_same_thread=False)
            self.db.execute("CREATE TABLE IF NOT EXISTS pairs (key TEXT PRIMARY KEY, score REAL, ms REAL)")

    def _key(self, query: str, doc_id: str) -> str:
        return hashlib.sha256(json.dumps([self.model_name, query, doc_id]).encode()).hexdigest()

    def score(self, query: str, doc_ids: list[str], corpus: dict[str, str]) -> list[float]:
        keys = [self._key(query, d) for d in doc_ids]
        got: dict[str, tuple[float, float]] = {}
        if self.db is not None:
            with self._lock:
                for i in range(0, len(keys), 500):
                    chunk = keys[i:i + 500]
                    q = f"SELECT key, score, ms FROM pairs WHERE key IN ({','.join('?' * len(chunk))})"
                    got.update({k: (s, ms) for k, s, ms in self.db.execute(q, chunk)})
        todo = [i for i, k in enumerate(keys) if k not in got]
        if todo:
            t0 = time.perf_counter()
            scores = self.model.predict([(query, corpus[doc_ids[i]]) for i in todo], batch_size=self.batch_size,
                                        show_progress_bar=False)
            per_pair = (time.perf_counter() - t0) * 1000 / len(todo)
            new = {keys[i]: (float(s), per_pair) for i, s in zip(todo, np.asarray(scores).reshape(-1))}
            got.update(new)
            if self.db is not None:
                with self._lock:
                    self.db.executemany("INSERT OR REPLACE INTO pairs VALUES (?, ?, ?)",
                                        [(k, s, ms) for k, (s, ms) in new.items()])
                    self.db.commit()
        self.last_latency_ms = sum(got[k][1] for k in keys)
        self.last_computed = len(todo)
        return [got[k][0] for k in keys]

    def rerank(self, query: str, hits: Hits, corpus: dict[str, str], k: int) -> Hits:
        if not hits:
            self.last_latency_ms, self.last_computed = 0.0, 0
            return hits
        scores = self.score(query, [d for d, _ in hits], corpus)
        order = np.argsort(-np.asarray(scores), kind="stable")[:k]
        return [(hits[i][0], float(scores[i])) for i in order]
