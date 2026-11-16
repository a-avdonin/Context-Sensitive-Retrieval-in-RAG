from arag.data import Task
from arag.llm import LLMResult
from arag.metrics import aggregate, per_query, records_to_run
from arag.pipeline import Strategy, all_strategies
from arag.queries import QueryBuilder
from arag.routers import LLMRouter, RuleRouter


class FakeLLM:
    def __init__(self, text):
        self.text, self.calls = text, []

    def chat(self, model, messages, max_tokens=512, **kw):
        self.calls.append(messages[-1]["content"])
        return LLMResult(self.text, 100, 10, 50.0)


def task(*user_turns):
    turns = []
    for u in user_turns:
        turns += [{"speaker": "user", "text": u}, {"speaker": "agent", "text": "answer"}]
    return Task("c<::>1", "fiqa", turns[:-1], gold_rewrite="gold")


def test_strategy_roundtrip():
    for s in all_strategies():
        assert str(Strategy.parse(s)) == s
    assert len(all_strategies()) == 24


def test_metrics():
    qrels = {"q": {"a": 1, "b": 1}}
    run = records_to_run([{"task_id": "q", "hits": [["a", 3.0], ["x", 2.0], ["b", 1.0]]}])
    agg = aggregate(per_query(qrels, run, (1, 3)))
    assert agg["recall_1"] == 0.5 and agg["recall_3"] == 1.0


def test_query_modes():
    t = task("What is NAV?", "Which is more important?")
    llm = FakeLLM("<think>x</think> Is NAV or market cap more important?")
    qb = QueryBuilder(llm, "m")
    assert qb.build(t, "last").text == "Which is more important?"
    assert qb.build(t, "gold_rewrite").text == "gold"
    q = qb.build(t, "llm_rewrite")
    assert q.text == "Is NAV or market cap more important?" and q.prompt_tokens == 100
    assert "What is NAV?" in llm.calls[0]


def test_rule_router():
    r = RuleRouter()
    assert r.route(task("What is the capital gains tax rate in the US?")).strategy == r.cheap
    assert r.route(task("What is NAV?", "How is it calculated?")).strategy == r.contextual
    assert r.route(task("What is NAV?", "enterprise value")).strategy == r.hard
    assert r.route(task("What is NAV?", "What does the SEC require for mutual fund NAV reporting?")).strategy == r.cheap


def test_llm_router_history_and_fallback():
    t = task("What is NAV?", "How is it calculated?")
    ok = LLMRouter(FakeLLM('{"rewrite": true, "retriever": "dense", "rerank": true, "reason": "pronoun"}'), "m", True)
    assert ok.route(t).strategy == "llm_rewrite:dense+rerank"
    nohist = LLMRouter(FakeLLM('{"rewrite": true, "retriever": "bm25", "rerank": false}'), "m", False)
    assert nohist.route(t).strategy == "last:bm25"
    assert "What is NAV?" not in nohist.llm.calls[0]
    bad = LLMRouter(FakeLLM("no json"), "m", True)
    assert bad.route(t).strategy == bad.fallback


def test_oracle_pick_and_deployable():
    from arag.cli import is_deployable, oracle_pick

    assert not is_deployable("gold_rewrite:bm25") and is_deployable("last:hybrid+rerank")
    scores = {"last:bm25": 0.5, "last:hybrid+rerank": 0.55, "llm_rewrite:dense": 0.55}
    assert oracle_pick(scores)[0] == "last:hybrid+rerank"  # ничья → дешевле
    assert oracle_pick(scores, eps=0.1)[0] == "last:bm25"


def test_stats():
    import numpy as np

    from arag.stats import holm, paired_bootstrap, randomization_test

    a = np.zeros(200)
    b = np.r_[np.ones(100), np.zeros(100)] * 0.2
    d, lo, hi = paired_bootstrap(a, b, 2000)
    assert abs(d - 0.1) < 1e-9 and lo > 0 and hi < 0.2
    assert randomization_test(a, b, 2000) < 0.01
    assert randomization_test(a, a) == 1.0
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_rewrite_stats_and_report():
    from arag.analysis import build_report, content_tokens, rewrite_stats

    assert content_tokens("What is the NAV of a fund?") == {"nav", "fund"}
    t = task("What is NAV of a mutual fund?", "How is it calculated?")
    t.gold_rewrite = "How is NAV of a mutual fund calculated?"
    st = rewrite_stats(t)
    assert st["deficit"] == 0.75 and st["missing_in_history"] == 1.0  # nav, mutual, fund из 4 токенов rewrite
    assert rewrite_stats(task("only first turn")) is None
    t.meta = {"multi_turn": "Follow-up", "answerability": "ANSWERABLE", "question_type": ["Factoid"]}
    sc = {"last:bm25": {t.task_id: {"ndcg_cut_5": 0.0}}, "gold_rewrite:bm25": {t.task_id: {"ndcg_cut_5": 1.0}}}
    rep = build_report([t], {t.task_id: {"d": 1}}, sc, "ndcg_cut_5", "toy")
    assert "## 5." in rep and "| [2/3, 1] | 1 | 0.000 | 1.000 | 1.000 | 1 | 0 |" in rep


def test_llm_helpers():
    from arag.llm import parallel_map, parse_json, strip_think

    assert strip_think("<think>a</think> hi") == "hi"
    assert strip_think("reasoning without open tag</think>\nanswer") == "answer"
    assert parse_json('x {"a": 1} y') == {"a": 1} and parse_json("{broken") == {} and parse_json("{a: 1}") == {}
    assert parallel_map(lambda x: x * 2, range(20), workers=4) == [x * 2 for x in range(20)]


def test_llm_cache_threadsafe(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from arag.llm import LLM

    llm = LLM("http://x/v1", "k", str(tmp_path), extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    calls = []

    def create(**kw):
        calls.append(kw)
        text, fr = ("", "length") if kw["messages"][0]["content"] == "empty" else ("ok", "stop")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=fr)],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1))

    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    from arag.llm import parallel_map

    res = parallel_map(lambda i: llm.chat("m", [{"role": "user", "content": str(i % 5)}]), range(40), workers=8)
    assert all(r.text == "ok" for r in res)
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    n = len(calls)
    assert llm.chat("m", [{"role": "user", "content": "1"}]).cached and len(calls) == n
    llm.chat("m", [{"role": "user", "content": "empty"}])
    llm.chat("m", [{"role": "user", "content": "empty"}])  # пустой обрезанный ответ не кэшируется
    assert len(calls) == n + 2


class FakeEmbedLLM:
    """Детерминированные 'эмбеддинги': мешок слов по хэшу в 32 измерения."""

    def __init__(self, fail_after=None):
        self.batches, self.fail_after = 0, fail_after

    @staticmethod
    def vec(text):
        import numpy as np

        v = np.zeros(32, dtype=np.float32)
        for w in text.lower().split():
            v[sum(map(ord, w)) % 32] += 1
        return v

    def embed_batch(self, model, texts):
        import numpy as np

        self.batches += 1
        if self.fail_after is not None and self.batches > self.fail_after:
            raise RuntimeError("server down")
        return np.stack([self.vec(t) for t in texts])

    def embed_one(self, model, text, use_cache=True):
        return self.vec(text), 12.5


def test_api_dense_index_resume_and_search(tmp_path):
    import pytest

    from arag.retrievers import APIDenseRetriever, api_index_path, build_api_index

    corpus = {f"d{i}": f"topic{i % 7} word{i}" for i in range(50)}
    corpus["nav"] = "net asset value nav fund"
    # первый прогон падает после 3 батчей → шарды сохранены, итогового файла нет
    with pytest.raises(RuntimeError):
        build_api_index(FakeEmbedLLM(fail_after=3), "m", corpus, tmp_path, batch_size=4, workers=1, shard_batches=1)
    assert not api_index_path(tmp_path, "m").exists()
    done = len(list(api_index_path(tmp_path, "m").with_suffix(".shards").glob("0*.npy")))
    assert done == 3
    llm = FakeEmbedLLM()
    build_api_index(llm, "m", corpus, tmp_path, batch_size=4, workers=1, shard_batches=1)
    assert llm.batches == 13 - 3  # 51 пассаж / 4 = 13 шардов, досчитаны только недостающие
    r = APIDenseRetriever(FakeEmbedLLM(), "m", corpus, tmp_path, batch_size=4)
    hits = r.search("nav fund", 3)
    assert hits[0][0] == "nav" and len(hits) == 3 and r.embed_ms == 12.5
    with pytest.raises(RuntimeError):  # индекс от другого корпуса не подхватывается молча
        APIDenseRetriever(FakeEmbedLLM(), "m", {"x": "y"}, tmp_path)


def test_engine_dense_api_latency(tmp_path):
    from arag.pipeline import Engine

    cfg = {"cache_dir": str(tmp_path), "data_dir": "", "retrieval": {"top_k": 2, "first_stage_k": 5, "rrf_k": 60},
           "dense": {"backend": "api", "model": "m", "batch_size": 4, "workers": 1},
           "llm": {"rewrite_model": "m", "max_history_turns": 6}}
    eng = Engine(cfg, FakeEmbedLLM())
    eng.corpus = lambda d: {"a": "net asset value fund", "b": "kubernetes pod", "c": "fund nav"}
    rec = eng.run(task("What is NAV fund?"), "last:dense")
    assert rec.latency_ms["embed"] == 12.5 and len(rec.hits) == 2
    assert {h[0] for h in rec.hits} <= {"a", "b", "c"}


class FakeCrossEncoder:
    """Оценка = число общих слов запроса и пассажа."""

    def __init__(self):
        self.pairs = 0

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        self.pairs += len(pairs)
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


def test_reranker_cache_and_latency(tmp_path):
    from arag.retrievers import Reranker

    corpus = {"a": "kubernetes pod", "b": "net asset value of a fund", "c": "fund nav value"}
    fake = FakeCrossEncoder()
    rr = Reranker("m", cache_path=tmp_path / "rr.sqlite", model=fake)
    hits = rr.rerank("nav value fund", [("a", 1.0), ("b", 0.9), ("c", 0.8)], corpus, 2)
    assert [h[0] for h in hits] == ["c", "b"] and fake.pairs == 3 and rr.last_computed == 3
    lat = rr.last_latency_ms
    # повтор и частично пересекающийся набор кандидатов: считаются только новые пары
    rr.rerank("nav value fund", [("c", 0), ("a", 0)], corpus, 2)
    assert fake.pairs == 3 and rr.last_computed == 0 and rr.last_latency_ms > 0
    corpus["d"] = "nav"
    rr.rerank("nav value fund", [("c", 0), ("d", 0)], corpus, 2)
    assert fake.pairs == 4 and rr.last_computed == 1
    # кэш переживает перезапуск; другая модель — другие ключи
    rr2 = Reranker("m", cache_path=tmp_path / "rr.sqlite", model=FakeCrossEncoder())
    rr2.rerank("nav value fund", [("a", 0), ("b", 0), ("c", 0)], corpus, 3)
    assert rr2.model.pairs == 0 and abs(rr2.last_latency_ms - lat) < 1e-6
    rr3 = Reranker("other", cache_path=tmp_path / "rr.sqlite", model=FakeCrossEncoder())
    rr3.rerank("nav value fund", [("a", 0)], corpus, 1)
    assert rr3.model.pairs == 1


def test_engine_rerank_records_model(tmp_path):
    from arag.pipeline import Engine
    from arag.retrievers import Reranker

    cfg = {"cache_dir": str(tmp_path), "data_dir": "", "retrieval": {"top_k": 2, "first_stage_k": 5, "rrf_k": 60},
           "dense": {"backend": "api", "model": "m", "batch_size": 4, "workers": 1},
           "reranker": {"model": "fake-ce", "candidates": 3},
           "llm": {"rewrite_model": "m", "max_history_turns": 6}}
    eng = Engine(cfg, FakeEmbedLLM())
    eng.corpus = lambda d: {"a": "net asset value fund", "b": "kubernetes pod", "c": "fund nav", "d": "x y"}
    eng._reranker = Reranker("fake-ce", model=FakeCrossEncoder())
    rec = eng.run(task("What is NAV fund?"), "last:dense+rerank")
    assert rec.reranker == "fake-ce" and rec.rerank_pairs == 3 and "rerank" in rec.latency_ms
    assert eng.run(task("What is NAV fund?"), "last:dense").reranker == ""
