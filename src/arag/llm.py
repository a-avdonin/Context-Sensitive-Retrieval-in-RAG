import hashlib
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    finish_reason: str = ""


class LLM:
    """OpenAI-совместимый клиент (LiteLLM, vLLM и т.п.) с кэшем на sqlite.

    Безопасен для вызова из нескольких потоков: запись в кэш под блокировкой.
    extra_body уходит в каждый запрос (например, отключение thinking у Qwen3) и входит в ключ кэша.
    """

    def __init__(self, base_url: str, api_key: str, cache_dir: str, temperature: float = 0.0,
                 timeout: float = 120.0, max_retries: int = 5, extra_body: dict | None = None):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.temperature = temperature
        self.extra_body = extra_body or {}
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(Path(cache_dir) / "llm_cache.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS emb_cache (key TEXT PRIMARY KEY, latency REAL, vec BLOB)")

    def _key(self, model, messages, max_tokens, kw) -> str:
        payload = [model, messages, max_tokens, self.temperature, kw]
        if self.extra_body:
            payload.append(self.extra_body)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def chat(self, model: str, messages: list[dict], max_tokens: int = 512, use_cache: bool = True,
             **kw) -> LLMResult:
        key = self._key(model, messages, max_tokens, kw)
        if use_cache:
            with self._lock:
                row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            if row:
                return LLMResult(**json.loads(row[0]) | {"cached": True})

        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=self.temperature,
            extra_body=self.extra_body or None, **kw
        )
        choice = resp.choices[0]
        res = LLMResult(
            text=choice.message.content or "",
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            finish_reason=choice.finish_reason or "",
        )
        # пустой ответ, обрезанный по длине (обычно thinking съел max_tokens), не кэшируем
        if use_cache and not (res.finish_reason == "length" and not strip_think(res.text)):
            with self._lock:
                self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)",
                                (key, json.dumps(res.__dict__ | {"cached": False})))
                self.db.commit()
        return res

    # ---------- эмбеддинги ----------

    def embed_batch(self, model: str, texts: list[str]):
        """Эмбеддинги пачки текстов без кэша (для корпуса). Возвращает np.ndarray [n, dim] float32."""
        import numpy as np

        resp = self.client.embeddings.create(model=model, input=texts)
        data = sorted(resp.data, key=lambda d: d.index)
        return np.asarray([d.embedding for d in data], dtype=np.float32)

    def embed_one(self, model: str, text: str, use_cache: bool = True):
        """Эмбеддинг одного запроса с кэшем. Возвращает (вектор float32, latency первого вызова, мс)."""
        import numpy as np

        key = hashlib.sha256(json.dumps(["emb", model, text]).encode()).hexdigest()
        if use_cache:
            with self._lock:
                row = self.db.execute("SELECT latency, vec FROM emb_cache WHERE key=?", (key,)).fetchone()
            if row:
                return np.frombuffer(row[1], dtype=np.float32).copy(), row[0]
        t0 = time.perf_counter()
        vec = self.embed_batch(model, [text])[0]
        latency = (time.perf_counter() - t0) * 1000
        if use_cache:
            with self._lock:
                self.db.execute("INSERT OR REPLACE INTO emb_cache VALUES (?, ?, ?)", (key, latency, vec.tobytes()))
                self.db.commit()
        return vec, latency

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)


def parallel_map(fn, items, workers: int = 8, desc: str = ""):
    """map с пулом потоков и прогресс-баром; порядок результатов сохраняется."""
    from tqdm import tqdm

    items = list(items)
    if workers <= 1:
        return [fn(x) for x in tqdm(items, desc=desc)]
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)  # открывающий тег съеден шаблоном чата
    return text.strip()


def parse_json(text: str) -> dict:
    text = strip_think(text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def format_history(turns: list[dict], max_turns: int | None = None) -> str:
    turns = turns[-max_turns:] if max_turns else turns
    return "\n".join(f"{t['speaker'].upper()}: {t['text'].strip()}" for t in turns)
