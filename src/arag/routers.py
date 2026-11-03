"""Роутеры выбирают стратегию retrieval для задачи.

fixed      — один пайплайн для всех запросов
rule       — эвристики по запросу и истории
llm_nohist — LLM видит только последний вопрос (аналог Adaptive-RAG: адаптация по сложности)
llm_hist   — LLM видит вопрос и историю, решает про rewrite, ретривер и reranking
"""
import re
from dataclasses import dataclass

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
    def __init__(self, cheap: str = "last:hybrid", contextual: str = "llm_rewrite:hybrid",
                 hard: str = "llm_rewrite:hybrid+rerank", short_len: int = 3):
        self.cheap, self.contextual, self.hard, self.short_len = cheap, contextual, hard, short_len

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
            return Route(self.cheap, rationale="first turn")
        dependent = f["anaphora"] or f["short"] or f["no_question"]
        if not dependent:
            return Route(self.cheap, rationale="standalone")
        return Route(self.hard if f["short"] else self.contextual, rationale=",".join(k for k, v in f.items() if v))


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
                 fallback: str = "last:hybrid"):
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


def build_router(name: str, cfg: dict, llm: LLM | None = None, fixed: str = "last:hybrid+rerank"):
    model, turns = cfg["llm"]["router_model"], cfg["llm"]["max_history_turns"]
    if name == "fixed":
        return FixedRouter(fixed)
    if name == "rule":
        return RuleRouter()
    if name == "llm_nohist":
        return LLMRouter(llm, model, use_history=False, max_history_turns=turns)
    if name == "llm_hist":
        return LLMRouter(llm, model, use_history=True, max_history_turns=turns)
    raise ValueError(name)
