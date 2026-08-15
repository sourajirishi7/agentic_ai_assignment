"""Retrieval-augmented answer generation."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings
from .vector_store import ScheduleVectorStore


class ScheduleRAG:
    """Simple modern RAG pipeline returning an answer and source documents."""

    def __init__(self, vector_store: ScheduleVectorStore, llm: object | None = None) -> None:
        self.vector_store = vector_store
        self.llm = llm or self._build_llm()

    def _build_llm(self) -> object | None:
        if not settings.openai_api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.llm_model, temperature=0)
        except Exception:
            return None

    def answer(self, query: str) -> dict[str, object]:
        documents = self.vector_store.query(query)
        context = "\n\n".join(doc.page_content for doc in documents)
        if self.llm is None:
            return {
                "answer": "Relevant schedule entries:\n" + (context or "No relevant schedule entries found."),
                "source_documents": documents,
            }

        messages = [
            SystemMessage(
                content=(
                    "You are a schedule RAG assistant. Answer only from the supplied schedule context. "
                    "If the context is empty or insufficient, say so."
                )
            ),
            HumanMessage(content=f"Question: {query}\n\nSchedule context:\n{context}"),
        ]
        try:
            response = self.llm.invoke(messages)  # type: ignore[attr-defined]
            answer = getattr(response, "content", str(response))
        except Exception as exc:
            answer = f"Unable to call the LLM for this RAG answer: {exc}"
        return {"answer": answer, "source_documents": documents}

