"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    data_dir: Path = PROJECT_ROOT / "data"
    schedule_path: Path = PROJECT_ROOT / "data" / "schedule.json"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    chroma_collection: str = "schedule_events"
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    llm_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    retriever_k: int = int(os.getenv("SCHEDULE_RETRIEVER_K", "8"))


settings = Settings()

