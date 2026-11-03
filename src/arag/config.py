import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    cfg = yaml.safe_load(open(path or ROOT / "config.yaml"))
    for key in ("data_dir", "cache_dir", "runs_dir"):
        p = Path(cfg[key])
        cfg[key] = str(p if p.is_absolute() else ROOT / p)
    cfg["llm"]["base_url"] = os.getenv("LLM_BASE_URL", cfg["llm"].get("base_url"))
    cfg["llm"]["api_key"] = os.getenv("LLM_API_KEY", "EMPTY")
    return cfg
