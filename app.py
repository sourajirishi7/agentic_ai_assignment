"""Render deployment entry point.

Exposes the FastAPI ``app`` defined in :mod:`src.web` so Render can
auto-discover it. On first import the agent (and therefore the schedule
data layer + ChromaDB vector store) is initialised.

Render start command typically used: ``uvicorn app:app --host 0.0.0.0 --port $PORT``
"""

from __future__ import annotations

import os

# Importing src.web instantiates `agent = ScheduleAgent()` at module level,
# which loads/validates data/schedule.json (or generates sample data) and
# syncs the ChromaDB vector store.
from src.web import app  # noqa: F401  (re-exported for uvicorn/gunicorn)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
