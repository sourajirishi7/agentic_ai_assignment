"""ChromaDB vector-store integration for schedule entries."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import ChromaError
from langchain_core.documents import Document

from .config import settings
from .models import ScheduleEntry


class LocalEmbeddingModel:
    """Embedding wrapper that prefers sentence-transformers and falls back offline."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self._model: Any | None = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return [list(vector) for vector in self._model.encode(texts, normalize_embeddings=True)]
        return [self._hash_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token.strip(".,:;!?()[]").lower() for token in text.split()]
        for token in tokens:
            if not token:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, len(digest), 2):
                index = int.from_bytes(digest[i : i + 2], "big") % self.dimensions
                vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def entry_to_document_text(entry: ScheduleEntry) -> str:
    event_date = entry.date.strftime("%B %d, %Y")
    start = datetime.combine(entry.date, entry.start_time).strftime("%I:%M %p").lstrip("0")
    end = datetime.combine(entry.date, entry.end_time).strftime("%I:%M %p").lstrip("0")
    return (
        f"{entry.title}.\n"
        f"Date: {event_date}.\n"
        f"Time: {start} to {end}.\n"
        f"Type: {entry.event_type.title()}.\n"
        f"Location: {entry.location}.\n"
        f"Description: {entry.description}\n"
        f"Status: {entry.status.title()}."
    )


class ScheduleVectorStore:
    """Persistent ChromaDB store synchronized with schedule JSON."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        embedding_model: LocalEmbeddingModel | None = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.chroma_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or LocalEmbeddingModel()
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=collection_name or settings.chroma_collection,
                metadata={"description": "Schedule assistant events"},
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to initialize ChromaDB: {exc}") from exc

    def sync(self, entries: list[ScheduleEntry]) -> None:
        """Ensure Chroma contains exactly the supplied schedule entries."""

        try:
            existing = self.collection.get(include=[])
            existing_ids = set(existing.get("ids", []))
            new_ids = {entry.id for entry in entries}
            stale_ids = list(existing_ids - new_ids)
            if stale_ids:
                self.collection.delete(ids=stale_ids)
            for entry in entries:
                self.upsert_entry(entry)
        except ChromaError as exc:
            raise RuntimeError(f"Unable to synchronize ChromaDB: {exc}") from exc

    def upsert_entry(self, entry: ScheduleEntry) -> None:
        text = entry_to_document_text(entry)
        metadata = {
            "event_id": entry.id,
            "date": entry.date.isoformat(),
            "start_time": entry.start_time.strftime("%H:%M"),
            "end_time": entry.end_time.strftime("%H:%M"),
            "event_type": entry.event_type,
            "status": entry.status,
        }
        embedding = self.embedding_model.embed_query(text)
        self.collection.upsert(ids=[entry.id], documents=[text], metadatas=[metadata], embeddings=[embedding])

    def delete_entry(self, event_id: str) -> None:
        self.collection.delete(ids=[event_id])

    def query(self, query: str, k: int | None = None, where: dict[str, Any] | None = None) -> list[Document]:
        try:
            result = self.collection.query(
                query_embeddings=[self.embedding_model.embed_query(query)],
                n_results=k or settings.retriever_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except ChromaError as exc:
            raise RuntimeError(f"Unable to query ChromaDB: {exc}") from exc

        documents: list[Document] = []
        ids = result.get("ids", [[]])[0]
        texts = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc_id, text, metadata, distance in zip(ids, texts, metadatas, distances):
            enriched = dict(metadata or {})
            enriched["id"] = doc_id
            enriched["distance"] = distance
            documents.append(Document(page_content=text or "", metadata=enriched))
        return documents

