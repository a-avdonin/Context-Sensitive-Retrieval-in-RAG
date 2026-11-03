import csv
import json
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]

SOURCES = {
    "mtrag": ("mtrag-human/generation_tasks/RAG.jsonl", "mtrag-human/retrieval_tasks/{d}/qrels/dev.tsv"),
    "mtragun": ("mtragun-human/generation_tasks/reference.jsonl", "mtragun-human/retrieval_tasks/qrels/{d}.tsv"),
}


def domain_of(collection: str) -> str:
    c = collection.lower()
    if "cloud" in c:
        return "cloud"
    for d in ("clapnq", "fiqa", "govt"):
        if d in c:
            return d
    raise ValueError(f"unknown collection {collection}")


@dataclass
class Task:
    task_id: str
    domain: str
    turns: list[dict]
    gold_rewrite: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def query(self) -> str:
        return self.turns[-1]["text"].strip()

    @property
    def history(self) -> list[dict]:
        return self.turns[:-1]

    @property
    def user_turn(self) -> int:
        return sum(t["speaker"] == "user" for t in self.turns)


@lru_cache(maxsize=4)
def load_corpus(root: str, domain: str) -> dict[str, str]:
    path = Path(root) / "corpora/passage_level" / f"{domain}.jsonl.zip"
    docs = {}
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            d = json.loads(line)
            title, text = (d.get("title") or "").strip(), d["text"].strip()
            docs[d["_id"]] = f"{title}\n{text}" if title and not text.startswith(title) else text
    return docs


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return qrels


def load_tasks(root: str, dataset: str = "mtrag", domains: list[str] | None = None):
    root_p, domains = Path(root), domains or DOMAINS
    src, qrels_tpl = SOURCES[dataset]
    qrels = {}
    for d in domains:
        qrels.update(load_qrels(root_p / qrels_tpl.format(d=d)))

    tasks = []
    with open(root_p / src) as f:
        for line in f:
            r = json.loads(line)
            d = domain_of(r["Collection"])
            if d not in domains:
                continue
            meta = {
                "answerability": (r.get("Answerability") or r.get("answerability") or [None])[0],
                "multi_turn": (r.get("Multi-Turn") or [None])[0],
                "question_type": r.get("Question Type") or [],
                "targets": [t["text"] for t in r.get("targets", [])],
                "has_qrels": r["task_id"] in qrels,
            }
            turns = [{"speaker": t["speaker"], "text": t["text"]} for t in r["input"]]
            tasks.append(Task(r["task_id"], d, turns, r.get("rewritten_query"), meta))
    return tasks, qrels
