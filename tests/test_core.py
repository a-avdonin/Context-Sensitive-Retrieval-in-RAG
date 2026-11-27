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
    assert len(all_strategies()) == 30  # 5 query modes × 3 retrievers × rerank


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
    assert r.route(task("What is the capital gains tax rate in the US?")).strategy == r.first_turn
    assert r.route(task("What is NAV?", "How is it calculated?")).strategy == r.dependent
    assert r.route(task("What is NAV?", "enterprise value")).strategy == r.dependent
    assert r.route(task("What is NAV?", "What does the SEC require for mutual fund NAV reporting?")).strategy \
        == r.standalone
    custom = RuleRouter(first_turn="last:bm25", standalone="last:dense", dependent="llm_rewrite:dense")
    assert custom.route(task("What is NAV?", "How is it calculated?")).strategy == "llm_rewrite:dense"


def test_oracle_router(tmp_path):
    import json

    from arag.routers import OracleRouter

    p = tmp_path / "o.jsonl"
    p.write_text(json.dumps({"task_id": "c<::>1", "best": "llm_rewrite:dense+rerank", "best_cheap": "last:dense"}))
    t = task("What is NAV?", "How is it calculated?")
    assert OracleRouter(p).route(t).strategy == "llm_rewrite:dense+rerank"
    assert OracleRouter(p, "best_cheap").route(t).strategy == "last:dense"
    t.task_id = "other"
    assert OracleRouter(p).route(t).rationale == "no_label"


def test_llm_router_history_and_fallback():
    from arag.config import load_config
    from arag.routers import build_router

    t = task("What is NAV?", "How is it calculated?")
    cfg = load_config()
    reply = '{"rewrite": true, "retriever": "dense", "rerank": true, "reason": "pronoun"}'
    ok = build_router("llm_hist", cfg, FakeLLM(reply))
    assert ok.route(t).strategy == "llm_rewrite:dense+rerank"
    control = build_router("llm_nohist_rewrite", cfg, FakeLLM(reply))
    assert control.route(t).strategy == "llm_rewrite:dense+rerank"
    assert control.llm.calls[0] == ok.llm.calls[0].replace("USER: What is NAV?\nAGENT: answer", "(empty)")
    first_turn = task("What is NAV?")
    ok.route(first_turn)
    control.route(first_turn)
    assert control.llm.calls[1] == ok.llm.calls[1]
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


def test_routes_summary(tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    import arag.cli as cli

    t1, t2 = task("What is NAV?"), task("What is NAV?", "How is it calculated?")
    t1.task_id, t2.task_id = "c<::>1", "c<::>2"
    qrels = {"c<::>1": {"a": 1}, "c<::>2": {"b": 1}}
    monkeypatch.setattr(cli, "_tasks", lambda cfg, args: ([t1, t2], qrels))
    d = tmp_path / "mtrag_routes"
    d.mkdir()
    lat = {"total": 100.0, "router": 10.0}
    recs = [{"task_id": "c<::>1", "route": "last:dense+rerank", "hits": [["a", 1.0]], "latency_ms": lat,
             "prompt_tokens": 50, "completion_tokens": 5, "rerank_pairs": 50, "rationale": "x"},
            {"task_id": "c<::>2", "route": "llm_rewrite:dense", "hits": [["x", 1.0]], "latency_ms": lat,
             "prompt_tokens": 450, "completion_tokens": 15, "rerank_pairs": 0, "rationale": "parse_error"}]
    (d / "rule.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    cfg = {"runs_dir": str(tmp_path), "eval": {"k_values": [1, 3, 5, 10]}}
    cli.cmd_routes_summary(cfg, SimpleNamespace(dataset="mtrag", slice="all"))
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if l.startswith("rule"))
    cells = row.split("\t")
    assert cells[2] == "0.5000"            # nDCG@5: один запрос найден, другой нет
    assert cells[4] == "1.0000"            # rewrite_rate среди later-задач
    assert cells[5] == "0.5000" and cells[-1] == "1"


class ChiqFakeLLM:
    """Отвечает по типу шага CHIQ (узнаёт его по инструкции)."""

    def __init__(self, switch=False, rewrite='{"query": "NAV vs market cap importance"}',
                 pr="Market cap matters more for stocks."):
        self.switch, self.rewrite, self.pr, self.calls = switch, rewrite, pr, []

    def chat(self, model, messages, max_tokens=512, **kw):
        p = messages[-1]["content"]
        self.calls.append(p)
        if "introduces a new topic" in p:
            out = "new_topic" if self.switch else "old_topic"
        elif "ambiguous" in p:
            out = "Which is more important, NAV or market cap?"
        elif "long version of the answer" in p:
            out = "NAV is the per-share value of fund assets minus liabilities."
        elif "one sentence response" in p:
            out = self.pr
        elif "summarizes the information" in p:
            out = "The user asked about NAV and market cap."
        else:
            out = self.rewrite
        return LLMResult(out, 10, 2, 5.0)


def test_chiq_pipeline():
    from arag.queries import qa_pairs

    t = task("What is NAV?", "And market cap?", "Which one is more important?")
    assert qa_pairs(t.history) == [("What is NAV?", "answer"), ("And market cap?", "answer")]
    llm = ChiqFakeLLM()
    q = QueryBuilder(llm, "m").build(t, "chiq")
    assert q.info["calls"] == 6 and not q.info["topic_switch"]           # TS, QD, RE, PR, HS, rewrite
    assert q.rerank_text == "NAV vs market cap importance"
    assert q.text.startswith("NAV vs market cap importance Market cap matters")
    assert q.text.endswith("Which is more important, NAV or market cap?")
    assert q.prompt_tokens == 60 and q.latency_ms == 30.0
    assert "### Context" in llm.calls[0] and "### Question" in llm.calls[0]
    # смена темы: история урезается до последней пары, summary не вызывается
    llm2 = ChiqFakeLLM(switch=True, rewrite="plain rewrite without json")
    q2 = QueryBuilder(llm2, "m").build(t, "chiq")
    assert q2.info["calls"] == 5 and q2.info["topic_switch"] and q2.rerank_text == "plain rewrite without json"
    assert "What is NAV?" not in llm2.calls[-1]
    # первый ход: без вызовов LLM
    assert QueryBuilder(ChiqFakeLLM(), "m").build(task("What is NAV?"), "chiq").text == "What is NAV?"


def test_engine_uses_rerank_text(tmp_path):
    from arag.pipeline import Engine
    from arag.retrievers import Reranker

    seen = []

    class SpyCE(FakeCrossEncoder):
        def predict(self, pairs, **kw):
            seen.extend(q for q, _ in pairs)
            return super().predict(pairs, **kw)

    cfg = {"cache_dir": str(tmp_path), "data_dir": "", "retrieval": {"top_k": 2, "first_stage_k": 5, "rrf_k": 60},
           "dense": {"backend": "api", "model": "m", "batch_size": 4, "workers": 1},
           "reranker": {"model": "ce", "candidates": 3}, "llm": {"rewrite_model": "m", "max_history_turns": 6}}

    class Both(FakeEmbedLLM, ChiqFakeLLM):
        def __init__(self):
            FakeEmbedLLM.__init__(self)
            ChiqFakeLLM.__init__(self)

    eng = Engine(cfg, Both())
    eng.corpus = lambda d: {"a": "nav market cap", "b": "kubernetes pod", "c": "fund nav", "d": "x"}
    eng._reranker = Reranker("ce", model=SpyCE())
    rec = eng.run(task("What is NAV?", "Which one is more important?"), "chiq:dense+rerank")
    assert set(seen) == {"NAV vs market cap importance"}
    assert rec.query.startswith("NAV vs market cap importance") and rec.query_info["calls"] == 6


def test_chiq_drops_refusal_pseudo_response():
    from arag.queries import is_refusal

    assert is_refusal("The provided context does not contain information about NAV.")
    assert is_refusal("I'm sorry, but I don't have the answer to your question.")
    assert not is_refusal("Market cap matters more for stocks.")
    t = task("What is NAV?", "Which one is more important?")
    q = QueryBuilder(ChiqFakeLLM(pr="The provided context does not state this."), "m").build(t, "chiq")
    assert "provided context" not in q.text and not q.info["pseudo_response_used"]
    assert q.text == "NAV vs market cap importance Which is more important, NAV or market cap?"


def test_adaptive_rag_router(tmp_path, monkeypatch):
    import json

    import numpy as np

    from arag.data import Task
    from arag.routers import AdaptiveRAGRouter

    assert AdaptiveRAGRouter.label([0.5, 0.5, 0.9]) == 2
    assert AdaptiveRAGRouter.label([0.6, 0.6, 0.6]) == 0          # ничья → самый простой
    assert AdaptiveRAGRouter.label([0.0, 0.0, 0.0]) is None       # нерешаемый → без метки
    assert AdaptiveRAGRouter.label([0.50, 0.52, 0.0], eps=0.05) == 0

    levels = ["last:dense", "last:dense+rerank", "llm_rewrite:dense+rerank"]
    # 10 диалогов × 6 ходов; уровень, который "выигрывает", определяется словом в вопросе
    words = ["alpha", "beta", "gamma"]
    tasks, qrels, runs = [], {}, {s: [] for s in levels}
    for c in range(10):
        for k in range(6):
            lvl = (c + k) % 3
            tid = f"conv{c}<::>{k + 1}"
            tasks.append(Task(tid, "fiqa", [{"speaker": "user", "text": f"{words[lvl]} question {c} {k}"}]))
            qrels[tid] = {"rel": 1}
            for i, s in enumerate(levels):
                hit = "rel" if i >= lvl else "junk"   # уровни ≥ lvl находят документ, ниже — нет
                runs[s].append({"task_id": tid, "hits": [[hit, 1.0]]})
    tasks.append(Task("conv99<::>1", "fiqa", [{"speaker": "user", "text": "delta unsolvable"}]))
    qrels["conv99<::>1"] = {"rel": 1}
    for s in levels:
        runs[s].append({"task_id": "conv99<::>1", "hits": [["junk", 1.0]]})
    d = tmp_path / "mtrag"
    d.mkdir()
    for s, recs in runs.items():
        (d / f"{s.replace(':', '__')}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))

    class WordEmb:
        def embed_one(self, model, text, use_cache=True):
            v = np.full(8, 0.01, dtype=np.float32)
            for i, w in enumerate(words + ["delta"]):
                if w in text:
                    v[i] = 1.0
            return v, 7.0

    r = AdaptiveRAGRouter(WordEmb(), "bge-m3", "", levels, tmp_path, "mtrag", folds=5, C=10.0, workers=1)
    rep = r.prepare(tasks, qrels)
    assert rep["label_dist"] == [20, 20, 20] and rep["unlabeled"] == 1
    assert rep["cv_accuracy"] == 1.0 and rep["majority_accuracy"] == 1 / 3
    route = r.route(tasks[2])                                  # conv0, ход 3 → gamma → уровень C
    assert route.strategy == "llm_rewrite:dense+rerank" and route.latency_ms == 7.0
    assert route.rationale.startswith("level=C")

    from arag import cli
    from arag.metrics import read_jsonl, write_jsonl

    holdout = Task("holdout<::>1", "fiqa", [{"speaker": "user", "text": "gamma test"}])
    target_runs = tmp_path / "mtragun"
    target_runs.mkdir()
    # Разметка теста даёт выигрыш уровню A. Роутер должен выбрать C по обучению на gamma из MTRAG.
    for i, level in enumerate(levels):
        write_jsonl(target_runs / f"{level.replace(':', '__')}.jsonl", [{
            "task_id": holdout.task_id, "hits": [["rel" if i == 0 else "junk", 1.0]],
            "latency_ms": {"total": 10.0}, "prompt_tokens": 0, "completion_tokens": 0,
        }])
    cfg = cli.load_config()
    cfg["runs_dir"] = str(tmp_path)
    cfg["llm"]["workers"] = 1
    for t in tasks:
        t.meta["has_qrels"] = True
    holdout.meta["has_qrels"] = True

    def load_fixture(_root, dataset, domains=None):
        return (tasks, qrels) if dataset == "mtrag" else ([holdout], {holdout.task_id: {"rel": 1}})

    monkeypatch.setattr(cli, "_llm", lambda _cfg: WordEmb())
    monkeypatch.setattr(cli, "load_tasks", load_fixture)
    monkeypatch.setattr(cli, "load_config", lambda _path: cfg)
    monkeypatch.setattr("sys.argv", [
        "arag", "route", "--router", "adaptive_rag", "--dataset", "mtragun", "--train-dataset", "mtrag",
    ])
    cv_path = tmp_path / "mtragun_routes" / "adaptive_rag.jsonl"
    cv_path.parent.mkdir()
    cv_path.write_text("existing CV result\n")
    cli.main()
    result = read_jsonl(tmp_path / "mtragun_routes" / "adaptive_rag_from_mtrag.jsonl")
    assert len(result) == 1 and result[0]["route"] == levels[2]
    assert result[0]["hits"] == [["junk", 1.0]]
    assert cv_path.read_text() == "existing CV result\n"


def test_judge_scores_and_prompt():
    from arag.generation import clean_scores, judge_texts

    assert clean_scores({"correctness": 5, "faithfulness": 4.0, "completeness": 1, "comment": "ok"}) == \
        {"correctness": 5, "faithfulness": 4, "completeness": 1, "comment": "ok"}
    assert clean_scores({"correctness": 5, "faithfulness": 4}) == {}            # неполный ответ
    assert clean_scores({"correctness": 7, "faithfulness": 4, "completeness": 1}) == {}
    assert clean_scores({"correctness": True, "faithfulness": 4, "completeness": 1}) == {}
    llm = FakeLLM('{"correctness": 4, "faithfulness": 5, "completeness": 3, "comment": "fine"}')
    turns = [{"speaker": "user", "text": "What is NAV?"}]
    scores, _ = judge_texts(llm, "m", turns, "NAV is net asset value.", "It is net asset value.", [])
    assert scores["correctness"] == 4 and "(no passages retrieved)" in llm.calls[0]
    assert "NAV is net asset value." in llm.calls[0] and "USER: What is NAV?" in llm.calls[0]
    assert judge_texts(FakeLLM("not json"), "m", turns, "r", "a", ["p"])[0] == {}


def test_calibration_report_and_answers_table(tmp_path):
    import json

    from arag.judging import answers_table, calibration_report, system_name

    items = []
    for k in range(12):
        h = 1 + k % 4
        for m in ("reference", "gpt-4o"):
            items.append({"task_id": f"t{k}", "model_id": m,
                          "human": {"faithfulness": h, "completeness": h, "appropriateness": h, "naturalness": 4,
                                    "win-rate": 25.0 * h},
                          "official": {"rb_llm": h / 4, "rl_f": 1 - h / 4, "rb_agg": 0.5},
                          "judge": {"correctness": h + 1, "faithfulness": h + 1, "completeness": 5 - h} if k else {}})
    rep = calibration_report(items, "judge-m")
    assert "| судья: faithfulness | 1.00 (1.00)" in rep        # монотонная связь с людьми
    assert "| судья: completeness | -1.00 (-1.00)" in rep
    assert "| MTRAG: rb_agg | — |" in rep                       # константа → корреляция не определена
    assert "Судья не вернул оценки: 2" in rep and "без reference-ответов (n = 12)" in rep

    assert system_name(tmp_path / "llm_rewrite__dense+rerank.jsonl") == "llm_rewrite:dense+rerank"
    assert system_name(tmp_path / "route_adaptive_rag.jsonl") == "route/adaptive_rag"
    recs = [{"task_id": "a", "answer": "NAV is net asset value.", "gen_tokens": 100, "judge_tokens": 50,
             "judge": {"correctness": 5, "faithfulness": 5, "completeness": 3}},
            {"task_id": "b", "answer": "I cannot answer: the passages do not contain this.", "gen_tokens": 300,
             "judge_tokens": 50, "judge": {}}]
    p = tmp_path / "route_rule.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    rows, per_task = answers_table([p], lambda tid: True)
    assert rows[0]["system"] == "route/rule" and rows[0]["n"] == 2 and rows[0]["judged"] == 1
    assert rows[0]["correctness"] == 5.0 and rows[0]["declined"] == 0.5 and rows[0]["gen_tokens"] == 200.0
    assert per_task == {"route/rule": {"a": recs[0]["judge"]}}


def test_cmd_answer_and_summary(tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    import arag.cli as cli
    import arag.data as data

    ts = []
    for k in range(6):
        t = task("What is NAV?") if k % 2 == 0 else task("What is NAV?", "How is it calculated?")
        t.task_id, t.meta = f"c{k}<::>{1 + k % 2}", {"answerability": "ANSWERABLE", "targets": ["NAV is ..."]}
        ts.append(t)
    monkeypatch.setattr(cli, "_tasks", lambda cfg, args: (ts, {}))
    monkeypatch.setattr(data, "load_corpus", lambda root, d: {"a": "net asset value", "b": "other"})

    class GenJudge:
        def chat(self, model, messages, max_tokens=512, **kw):
            if model == "judge":
                return LLMResult('{"correctness": 4, "faithfulness": 5, "completeness": 3}', 200, 20, 5.0)
            return LLMResult("NAV is net asset value.", 300, 10, 9.0)

    monkeypatch.setattr(cli, "_llm", lambda cfg: GenJudge())
    cfg = {"runs_dir": str(tmp_path), "data_dir": "", "eval": {"k_values": [1, 3, 5, 10]},
           "llm": {"generator_model": "gen", "judge_model": "judge", "max_history_turns": 6, "workers": 2}}
    (tmp_path / "mtrag").mkdir()
    (tmp_path / "mtrag_routes").mkdir()
    run = tmp_path / "mtrag" / "last__dense.jsonl"
    route = tmp_path / "mtrag_routes" / "rule.jsonl"
    for p in (run, route):
        p.write_text("\n".join(json.dumps({"task_id": t.task_id, "hits": [["a", 1.0], ["b", 0.5]]}) for t in ts))
    args = SimpleNamespace(dataset="mtrag", run=[str(run), str(route)], k=1, sample=4, seed=0, workers=None)
    cli.cmd_answer(cfg, args)
    a = [json.loads(l) for l in open(tmp_path / "mtrag_answers" / "last__dense.jsonl")]
    b = [json.loads(l) for l in open(tmp_path / "mtrag_answers" / "route_rule.jsonl")]
    assert len(a) == 4 and {r["task_id"] for r in a} == {r["task_id"] for r in b}   # одна выборка для всех
    assert a[0]["docs"] == ["a"] and a[0]["judge"]["correctness"] == 4 and a[0]["gen_tokens"] == 310
    assert not list((tmp_path / "mtrag").glob("*answers*"))                          # run-папка не засоряется
    capsys.readouterr()
    cli.cmd_answers_summary(cfg, SimpleNamespace(dataset="mtrag", slice="all", baseline="last:dense",
                                                 metric="correctness", n=200))
    out = capsys.readouterr().out
    assert "route/rule\t4\t4\t4.0000\t5.0000\t3.0000" in out and "baseline=last:dense" in out


def test_transfer_holds_out_shared_openings_and_ids():
    from arag.data import held_out_dialogues

    source = task("What is NAV?")
    copied = task("  WHAT is   NAV? ", "How is it calculated?")
    copied.task_id = "copy<::>2"
    shared_id = task("Another opening")
    unseen = task("What is APR?")
    unseen.task_id = "new<::>1"
    assert held_out_dialogues([source], [copied, shared_id, unseen]) == [unseen]


def test_paired_tests_resample_whole_dialogues():
    import numpy as np

    from arag.stats import paired_bootstrap, randomization_test

    # При двух кластерах возможны эффекты 0, 1/4 и 1. Ходы имеют одинаковый вес.
    base = np.zeros(4)
    scores = np.array([0, 0, 0, 1])
    groups = ["a", "a", "a", "b"]
    assert paired_bootstrap(base, scores, groups=groups) == (0.25, 0.0, 1.0)
    # Все три выигрыша в одном диалоге меняют знак одновременно.
    assert randomization_test(base, [1, 1, 1, 0], groups=groups) == 1


def test_rewrite_report_assigns_exact_thirds():
    from arag.analysis import build_report

    tasks = []
    for i, query in enumerate(["apple orange pear", "apple orange", "apple", "mango"]):
        t = task("Earlier question", query)
        t.task_id, t.gold_rewrite = str(i), "apple orange pear"
        tasks.append(t)
    qrels = {t.task_id: {"d": 1} for t in tasks}
    scores = {t.task_id: {"ndcg_cut_5": 1.0} for t in tasks}
    report = build_report(tasks, qrels, {"last:bm25": scores, "gold_rewrite:bm25": scores}, "ndcg_cut_5", "test")
    bins = report.split("### Выигрыш от rewrite")[-1]
    assert "| 0 | 1 |" in bins and "| (0, 1/3) |" not in bins
    assert "| [1/3, 2/3) | 1 |" in bins and "| [2/3, 1] | 2 |" in bins
