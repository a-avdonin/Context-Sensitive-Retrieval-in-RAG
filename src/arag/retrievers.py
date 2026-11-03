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


class DenseRetriever:
    def __init__(self, corpus: dict[str, str], cache_dir: Path, model_name: str, query_prefix: str = "",
                 batch_size: int = 128, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = query_prefix
        self.ids = list(corpus)
        path = cache_dir / f"dense_{model_name.replace('/', '__')}.npy"
        if path.exists():
            self.emb = np.load(path)
        else:
            self.emb = self.model.encode(list(corpus.values()), batch_size=batch_size, normalize_embeddings=True,
                                         show_progress_bar=True, convert_to_numpy=True).astype(np.float16)
            np.save(path, self.emb)

    def search(self, query: str, k: int) -> Hits:
        q = self.model.encode([self.prefix + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.emb @ q.astype(np.float16)
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]


def rrf(runs: list[Hits], k: int, rrf_k: int = 60) -> Hits:
    fused: dict[str, float] = {}
    for hits in runs:
        for rank, (doc, _) in enumerate(hits):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]


class Reranker:
    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, hits: Hits, corpus: dict[str, str], k: int) -> Hits:
        if not hits:
            return hits
        scores = self.model.predict([(query, corpus[d]) for d, _ in hits], batch_size=self.batch_size)
        order = np.argsort(-np.asarray(scores))[:k]
        return [(hits[i][0], float(scores[i])) for i in order]
