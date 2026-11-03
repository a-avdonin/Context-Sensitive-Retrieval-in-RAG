"""Формирование поискового запроса из диалога."""
from dataclasses import dataclass

from .data import Task
from .llm import LLM, format_history, strip_think

REWRITE_PROMPT = """Rewrite the user's last question into a standalone search query.
Resolve pronouns and ellipsis using the conversation. Keep all constraints from the question.
Do not answer the question. Output only the rewritten query.

Conversation:
{history}

Last question: {query}
Standalone query:"""


@dataclass
class Query:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class QueryBuilder:
    def __init__(self, llm: LLM | None = None, model: str | None = None, max_history_turns: int = 6):
        self.llm, self.model, self.max_turns = llm, model, max_history_turns

    def build(self, task: Task, mode: str) -> Query:
        if mode == "last":
            return Query(task.query)
        if mode == "user_concat":
            users = [t["text"].strip() for t in task.turns if t["speaker"] == "user"]
            return Query(" ".join(users[-3:]))
        if mode == "gold_rewrite":
            return Query(task.gold_rewrite or task.query)
        if mode == "llm_rewrite":
            return self._llm_rewrite(task)
        raise ValueError(f"unknown query mode {mode}")

    def _llm_rewrite(self, task: Task) -> Query:
        if not task.history:
            return Query(task.query)
        prompt = REWRITE_PROMPT.format(history=format_history(task.history, self.max_turns), query=task.query)
        r = self.llm.chat(self.model, [{"role": "user", "content": prompt}], max_tokens=128)
        text = strip_think(r.text).strip().strip('"') or task.query
        return Query(text, r.prompt_tokens, r.completion_tokens, r.latency_ms)


QUERY_MODES = ["last", "user_concat", "gold_rewrite", "llm_rewrite"]
