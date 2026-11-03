"""Генерация ответа по найденным пассажам и оценка LLM-as-a-Judge."""
from .data import Task
from .llm import LLM, format_history, parse_json, strip_think

GEN_PROMPT = """Answer the user's last question using only the passages below.
If the passages do not contain the answer, say that you cannot answer from the available information.
Be concise.

Passages:
{passages}

Conversation:
{history}"""

JUDGE_PROMPT = """You evaluate an answer in a conversational RAG system.

Conversation:
{history}

Retrieved passages:
{passages}

Reference answer:
{reference}

Candidate answer:
{answer}

Score the candidate from 1 to 5 on each criterion:
- correctness: agrees with the reference answer, no factual errors
- faithfulness: every claim is supported by the retrieved passages
- completeness: covers all key points of the reference answer

Answer with JSON only: {{"correctness": int, "faithfulness": int, "completeness": int, "comment": "short"}}"""


def _passages(doc_ids: list[str], corpus: dict[str, str], max_chars: int = 1500) -> str:
    return "\n\n".join(f"[{i + 1}] {corpus[d][:max_chars]}" for i, d in enumerate(doc_ids))


def generate(llm: LLM, model: str, task: Task, doc_ids: list[str], corpus: dict[str, str], max_turns: int = 6):
    prompt = GEN_PROMPT.format(passages=_passages(doc_ids, corpus), history=format_history(task.turns, max_turns))
    r = llm.chat(model, [{"role": "user", "content": prompt}], max_tokens=512)
    return strip_think(r.text), r


def judge(llm: LLM, model: str, task: Task, answer: str, doc_ids: list[str], corpus: dict[str, str],
          max_turns: int = 6) -> tuple[dict, object]:
    prompt = JUDGE_PROMPT.format(
        history=format_history(task.turns, max_turns),
        passages=_passages(doc_ids, corpus, 800),
        reference=(task.meta.get("targets") or [""])[0],
        answer=answer,
    )
    r = llm.chat(model, [{"role": "user", "content": prompt}], max_tokens=300)
    return parse_json(r.text), r
