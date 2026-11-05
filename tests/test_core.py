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
