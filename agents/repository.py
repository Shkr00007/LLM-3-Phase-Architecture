from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from agents.base import BaseAgent
from database.db_manager import DatabaseManager


class TripletExtraction(BaseModel):
    triplets: list[dict[str, Any]] = Field(default_factory=list)


class MemorySummary(BaseModel):
    summary: str


@dataclass
class RetrievalBundle:
    summary: str | None
    chunks: list[dict[str, Any]]
    triplets: list[dict[str, Any]]


class RepositoryAgent(BaseAgent):
    def __init__(
        self,
        *,
        host: str,
        model: str,
        embed_model: str,
        db: DatabaseManager,
        keep_alive: str,
        headers: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> None:
        super().__init__(
            name="repository",
            model=model,
            host=host,
            keep_alive=keep_alive,
            headers=headers,
        )
        self.embed_model = embed_model
        self.db = db
        self.top_k = top_k

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 75) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
        return chunks

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        numerator = sum(x * y for x, y in zip(a, b))
        denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return 0.0 if denom == 0 else numerator / denom

    @staticmethod
    def normalize_terms(text: str) -> list[str]:
        return sorted({token.lower() for token in re.findall(r"[A-Za-z0-9_-]{3,}", text)})

    def ingest(self, *, source: str, content: str) -> dict[str, Any]:
        document_id = self.db.insert_document(source=source, content=content)
        chunks = self.chunk_text(content)
        for index, chunk in enumerate(chunks):
            embedding = self.embed(chunk, model=self.embed_model)
            self.db.insert_chunk(
                document_id=document_id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
                metadata={"source": source},
            )

        extraction = self.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract factual subject-predicate-object triplets from the text. "
                        "Use compact canonical entities and include a weight from 0.1 to 1.0."
                    ),
                },
                {"role": "user", "content": content[:8000]},
            ],
            schema=TripletExtraction.model_json_schema(),
            options={"temperature": 0},
        )
        parsed = TripletExtraction.model_validate(extraction)
        self.db.insert_triplets(document_id, parsed.triplets)
        return {
            "document_id": document_id,
            "chunks_indexed": len(chunks),
            "triplets_indexed": len(parsed.triplets),
        }

    def retrieve(self, *, session_id: str, query: str) -> RetrievalBundle:
        query_embedding = self.embed(query, model=self.embed_model)
        scored_chunks: list[dict[str, Any]] = []
        for row in self.db.fetch_chunks():
            embedding = json.loads(row["embedding"])
            score = self.cosine_similarity(query_embedding, embedding)
            scored_chunks.append(
                {
                    "chunk_id": row["id"],
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"]),
                    "score": score,
                }
            )
        top_chunks = sorted(scored_chunks, key=lambda item: item["score"], reverse=True)[: self.top_k]

        graph_rows = self.db.fetch_triplets_for_terms(self.normalize_terms(query))
        triplets = [dict(row) for row in graph_rows]
        summary = self.db.fetch_summary(session_id)
        return RetrievalBundle(summary=summary, chunks=top_chunks, triplets=triplets)

    def remember_turn(self, session_id: str, role: str, content: str) -> None:
        self.db.append_turn(session_id, role, content)

    def prune_context(self, *, session_id: str, rolling_window: int = 6) -> dict[str, Any]:
        turns = self.db.fetch_turns(session_id)
        if len(turns) <= rolling_window:
            return {"summary_updated": False, "window_turns": [dict(row) for row in turns]}

        stale_turns = turns[:-rolling_window]
        stale_text = "\n".join(f"{row['role']}: {row['content']}" for row in stale_turns)
        summary_result = self.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the conversation into mid-term memory. Preserve goals, decisions, facts, "
                        "constraints, and unresolved questions."
                    ),
                },
                {"role": "user", "content": stale_text[:12000]},
            ],
            schema=MemorySummary.model_json_schema(),
            options={"temperature": 0.1},
        )
        summary = MemorySummary.model_validate(summary_result).summary
        existing = self.db.fetch_summary(session_id)
        if existing:
            summary = f"{existing}\n{summary}".strip()
        self.db.upsert_summary(session_id, summary)
        return {"summary_updated": True, "window_turns": [dict(row) for row in turns[-rolling_window:]]}
