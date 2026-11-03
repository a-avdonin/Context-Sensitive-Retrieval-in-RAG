import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cached: bool = False


class LLM:
    """OpenAI-совместимый клиент (vLLM и т.п.) с кэшем на sqlite."""

    def __init__(self, base_url: str, api_key: str, cache_dir: str, temperature: float = 0.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(Path(cache_dir) / "llm_cache.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def chat(self, model: str, messages: list[dict], max_tokens: int = 512, **kw) -> LLMResult:
        key = hashlib.sha256(json.dumps([model, messages, max_tokens, self.temperature, kw], sort_keys=True).encode()).hexdigest()
        row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            return LLMResult(**json.loads(row[0]) | {"cached": True})

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=self.temperature, **kw
        )
        res = LLMResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(res.__dict__ | {"cached": False})))
        self.db.commit()
        return res


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict:
    text = strip_think(text)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def format_history(turns: list[dict], max_turns: int | None = None) -> str:
    turns = turns[-max_turns:] if max_turns else turns
    return "\n".join(f"{t['speaker'].upper()}: {t['text'].strip()}" for t in turns)
