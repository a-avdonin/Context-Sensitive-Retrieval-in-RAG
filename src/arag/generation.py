"""Генерация ответа по найденным пассажам и оценка LLM-as-a-Judge."""
from .data import Task
from .llm import LLM, format_history, parse_json, strip_think

CRITERIA = ("correctness", "faithfulness", "completeness")
PASSAGE_CHARS = 1500  # генератор и судья видят один и тот же текст пассажей

GEN_PROMPT = """Answer the user's last question using only the passages below.
If the passages do not contain the answer, say that you cannot answer from the available information.
Be concise.

Passages:
{passages}

Conversation:
{history}"""

JUDGE_PROMPT = """You evaluate an answer produced by a conversational question-answering system. The system must answer the user's last question using only the retrieved passages.

Conversation (the last USER turn is the question):
{history}

Retrieved passages:
{passages}

Reference answer (written by a human expert):
{reference}

Candidate answer:
{answer}

Rate the candidate on three criteria, each with an integer from 1 to 5.
- correctness: is the candidate factually consistent with the reference answer? 5 = fully agrees, no errors; 3 = partly right, or right but misses the main point; 1 = contradicts the reference or answers a different question. If the reference says the information is not available, a candidate that also declines is correct and a candidate that answers anyway is not.
- faithfulness: is every claim of the candidate supported by the retrieved passages? 5 = all claims supported; 3 = about half supported; 1 = mostly unsupported or contradicted. A candidate that only says it cannot answer is fully faithful.
- completeness: does the candidate cover the key points of the reference answer? 5 = all of them; 3 = about half; 1 = none.

Answer with JSON only: {{"correctness": int, "faithfulness": int, "completeness": int, "comment": "one short sentence"}}"""


def _passages(texts: list[str], max_chars: int) -> str:
    return "\n\n".join(f"[{i + 1}] {t[:max_chars]}" for i, t in enumerate(texts)) or "(no passages retrieved)"


def generate(llm: LLM, model: str, task: Task, doc_ids: list[str], corpus: dict[str, str], max_turns: int = 6):
    prompt = GEN_PROMPT.format(passages=_passages([corpus[d] for d in doc_ids], PASSAGE_CHARS),
                               history=format_history(task.turns, max_turns))
    r = llm.chat(model, [{"role": "user", "content": prompt}], max_tokens=512)
    return strip_think(r.text), r


def clean_scores(d: dict) -> dict:
    """Оставляет только корректные оценки 1..5; при неполном ответе судьи возвращает {} (не учитывается)."""
    out = {}
    for c in CRITERIA:
        v = d.get(c)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return {}
        out[c] = int(round(v))
    return out | {"comment": str(d.get("comment", ""))[:300]}


def judge_texts(llm: LLM, model: str, turns: list[dict], reference: str, answer: str, passages: list[str],
                max_turns: int = 6) -> tuple[dict, object]:
    """Судья по готовым текстам: используется и для наших ответов, и для калибровки на human-оценках."""
    prompt = JUDGE_PROMPT.format(history=format_history(turns, max_turns), passages=_passages(passages, PASSAGE_CHARS),
                                 reference=reference.strip() or "(no reference)", answer=answer.strip() or "(empty)")
    r = llm.chat(model, [{"role": "user", "content": prompt}], max_tokens=300)
    return clean_scores(parse_json(r.text)), r


def judge(llm: LLM, model: str, task: Task, answer: str, doc_ids: list[str], corpus: dict[str, str],
          max_turns: int = 6) -> tuple[dict, object]:
    return judge_texts(llm, model, task.turns, (task.meta.get("targets") or [""])[0], answer,
                       [corpus[d] for d in doc_ids], max_turns)
