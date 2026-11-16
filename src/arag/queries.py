"""Формирование поискового запроса из диалога."""
import re
from dataclasses import dataclass, field

from .data import Task
from .llm import LLM, format_history, parse_json, strip_think

REWRITE_PROMPT = """Rewrite the user's last question into a standalone search query.
Resolve pronouns and ellipsis using the conversation. Keep all constraints from the question.
Do not answer the question. Output only the rewritten query.

Conversation:
{history}

Last question: {query}
Standalone query:"""

# ---------- CHIQ-AD (Mo et al., EMNLP 2024, arXiv 2406.05013) ----------
# Инструкции по PROMPT_DICT из github.com/fengranMark/CHIQ (generation/enhancement_generation.py).
# Дословно сверены explain_question, explain_response, single_response и начало search_query/summarize_context;
# new_topic восстановлен по описанию (ответ "new_topic"/"old_topic"). Формат ввода оттуда же:
# инструкция, "### Context" со строками "вопрос ответ", "### Question".
CHIQ_PROMPTS = {
    # Topic Switch: новая тема → от истории остаётся только последняя пара
    "new_topic": "You are given a set of question-answer pairs and a new question. Your goal is to determine whether "
                 "the new question continues the discussion on an existing topic or introduces a new topic. "
                 "Answer only \"new_topic\" or \"old_topic\".",
    # Question Disambiguation: самодостаточная версия текущего вопроса
    "explain_question": "You are given a set of question-answers pairs and a new question that is ambiguous. Your goal "
                        "is to re-write the question so it became clear. Just write the new question without any "
                        "introduction.",
    # Response Expansion: расширенная версия последнего ответа системы
    "explain_response": "You are given a question-and-answer pair, where the answer is not clear. Your goal is to write "
                        "a long version of the answer based on its given context. The generated answer should be one "
                        "sentence only and less than 20 words.",
    # Pseudo Response: короткий ответ на новый вопрос (расширение запроса)
    # отклонение от оригинала: qwen3.8 без этой фразы в ~половине случаев отвечает "context does not contain..."
    # (LLaMA-2 у авторов отвечает из своих знаний); псевдо-ответ — расширение запроса, а не grounded-ответ
    "single_response": "Given a series of question-and-answer pairs, along with a new question, your task is to give a "
                       "one sentence response to the new question. If the pairs do not contain the answer, answer "
                       "from your general knowledge; never say that the information is missing.",
    # History Summary: сжатая история (только если тема не сменилась)
    "summarize_context": "You are given a context in the form of question-answer pairs. Your goal is to write a "
                         "paragraph that summarizes the information in the context. The summary should be short with "
                         "one sentence for each question answer pair.",
    # финальное переписывание по улучшенной истории
    "search_query": "You are given a context in the form of question-answer pairs and a new question. Your goal is to "
                    "convert the new question into a search engine query that can be used to retrieve relevant "
                    "documents. The output should be placed in a JSON dictionary as follow: {\"query\": \"\"}",
}


REFUSAL = re.compile(
    r"(does not|doesn't|do not|don't|cannot|can't|not) (contain|state|provide|mention|include|specify|have|know|find|"
    r"answer)|not (mentioned|specified|available|provided)|i'?m sorry|unable to|no information", re.I)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.search(text))


@dataclass
class Query:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    rerank_text: str | None = None          # текст для reranker'а, если отличается от поискового
    info: dict = field(default_factory=dict)  # промежуточные результаты (для анализа)


def qa_pairs(turns: list[dict]) -> list[tuple[str, str]]:
    """История диалога → пары (вопрос пользователя, ответ агента)."""
    pairs, q = [], None
    for t in turns:
        if t["speaker"] == "user":
            if q is not None:
                pairs.append((q, ""))
            q = t["text"].strip()
        elif q is not None:
            pairs.append((q, t["text"].strip()))
            q = None
    if q is not None:
        pairs.append((q, ""))
    return pairs


class QueryBuilder:
    def __init__(self, llm: LLM | None = None, model: str | None = None, max_history_turns: int = 6,
                 chiq_answer_chars: int = 600):
        self.llm, self.model, self.max_turns = llm, model, max_history_turns
        self.chiq_answer_chars = chiq_answer_chars

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
        if mode == "chiq":
            return self._chiq(task)
        raise ValueError(f"unknown query mode {mode}")

    def _llm_rewrite(self, task: Task) -> Query:
        if not task.history:
            return Query(task.query)
        prompt = REWRITE_PROMPT.format(history=format_history(task.history, self.max_turns), query=task.query)
        r = self.llm.chat(self.model, [{"role": "user", "content": prompt}], max_tokens=128)
        text = strip_think(r.text).strip().strip('"') or task.query
        return Query(text, r.prompt_tokens, r.completion_tokens, r.latency_ms)

    # ---------- CHIQ-AD ----------

    def _chiq_call(self, acc: Query, kind: str, context: str, question: str | None, max_tokens: int) -> str:
        parts = [CHIQ_PROMPTS[kind], "", "### Context", context]
        if question is not None:
            parts += ["", "### Question", question]
        r = self.llm.chat(self.model, [{"role": "user", "content": "\n".join(parts)}], max_tokens=max_tokens)
        acc.prompt_tokens += r.prompt_tokens
        acc.completion_tokens += r.completion_tokens
        acc.latency_ms += r.latency_ms
        acc.info.setdefault("calls", 0)
        acc.info["calls"] += 1
        return strip_think(r.text).strip()

    def _fmt_pairs(self, pairs: list[tuple[str, str]]) -> str:
        n = self.chiq_answer_chars
        return "\n".join(f"{q} {a[:n]}".strip() for q, a in pairs) or "(empty)"

    def _chiq(self, task: Task) -> Query:
        """CHIQ-AD: TS → QD, RE, PR → (HS, если тема не сменилась) → переписывание по улучшенной истории.

        Поисковый запрос = rewrite + pseudo response + disambiguated question (как в прогонах авторов
        "search_query+single_response+explain_question"); для reranker'а — только rewrite.
        Вызовов LLM: 5 при смене темы, 6 без неё.
        """
        if not task.history:
            return Query(task.query)
        acc = Query("")
        pairs = qa_pairs(task.history)
        ctx = self._fmt_pairs(pairs)
        u = task.query

        ts = self._chiq_call(acc, "new_topic", ctx, u, 16).lower()
        switch = "new_topic" in ts or ("new" in ts and "old" not in ts)
        if switch:
            pairs = pairs[-1:]
            ctx = self._fmt_pairs(pairs)

        qd = self._chiq_call(acc, "explain_question", ctx, u, 96).strip('"') or u
        last_q, last_a = pairs[-1]
        re_ = self._chiq_call(acc, "explain_response", f"{last_q} {last_a[:self.chiq_answer_chars]}", None, 64) \
            if last_a else ""
        pr = self._chiq_call(acc, "single_response", ctx, u, 96)
        pr_used = pr if pr and not is_refusal(pr) else ""  # отказ не несёт информации для расширения запроса
        enhanced = pairs[:-1] + [(last_q, re_ or last_a)]
        h = self._fmt_pairs(enhanced)
        hs = ""
        if not switch:
            hs = self._chiq_call(acc, "summarize_context", h, None, 256)
            h = hs or h

        raw = self._chiq_call(acc, "search_query", h, u, 96)
        rewrite = (parse_json(raw).get("query") or "").strip() if "{" in raw else raw.strip().strip('"')
        rewrite = rewrite or qd
        acc.text = " ".join(x for x in (rewrite, pr_used, qd) if x)
        acc.rerank_text = rewrite
        acc.info.update({"topic_switch": switch, "disambiguated": qd, "expanded_response": re_,
                         "pseudo_response": pr, "pseudo_response_used": bool(pr_used), "summary": hs, "rewrite": rewrite})
        return acc


QUERY_MODES = ["last", "user_concat", "gold_rewrite", "llm_rewrite", "chiq"]
LLM_QUERY_MODES = {"llm_rewrite", "chiq"}
