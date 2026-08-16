"""Web server exposing the ScheduleAgent over HTTP with a local UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent import ScheduleAgent

app = FastAPI(title="Agentic RAG Schedule Assistant")
agent = ScheduleAgent()

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/api/chat")
async def chat(request: Request) -> JSONResponse:
    data = await request.json()
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return JSONResponse({"reply": ""})
    reply = agent.invoke(user_input) or ""
    return JSONResponse({"reply": reply})


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health_root() -> JSONResponse:
    return JSONResponse({"status": "healthy"})


# Serve the frontend (index.html) at "/"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.web:app", host="0.0.0.0", port=port, reload=True)
