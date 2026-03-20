from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from ollama import ResponseError
from pydantic import BaseModel, Field

from agents.base import BaseAgent
from database.db_manager import DatabaseManager


LOGGER = logging.getLogger(__name__)


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
        chroma_path: str = "database/chroma",
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
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.memory_collection = self.chroma.get_or_create_collection(name="agent_memory")
        LOGGER.debug("RepositoryAgent initialized with Chroma path=%s top_k=%s", chroma_path, top_k)

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
    def normalize_terms(text: str) -> list[str]:
        return sorted({token.lower() for token in re.findall(r"[A-Za-z0-9_-]{3,}", text)})

    @staticmethod
    def fallback_triplets(text: str) -> list[dict[str, Any]]:
        sentences = [part.strip() for part in re.split(r"[.!?]\s+", text) if part.strip()]
        triplets: list[dict[str, Any]] = []
        for sentence in sentences[:10]:
            match = re.match(r"([A-Z][\w\-/ ]+)\s+(is|are|has|uses|builds|supports|contains)\s+(.+)", sentence)
            if not match:
                continue
            subject, predicate, obj = match.groups()
            triplets.append(
                {
                    "subject": subject.strip(),
                    "predicate": predicate.strip(),
                    "object": obj.strip()[:250],
                    "weight": 0.4,
                }
            )
        return triplets

    def safe_embed(self, text: str) -> list[float]:
        try:
            embedding = self.embed(text, model=self.embed_model)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Embedding failed for text length=%s model=%s", len(text), self.embed_model)
            raise RuntimeError(
                "Failed to generate embeddings from Ollama. Ensure the remote Ollama host is reachable and "
                f"the embedding model '{self.embed_model}' is available."
            ) from exc
        LOGGER.debug("Generated embedding with %s dimensions", len(embedding))
        return embedding

    def extract_triplets(self, text: str) -> list[dict[str, Any]]:
        try:
            extraction = self.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract factual subject-predicate-object triplets from the text. "
                            "Use compact canonical entities and include a weight from 0.1 to 1.0."
                        ),
                    },
                    {"role": "user", "content": text[:8000]},
                ],
                schema=TripletExtraction.model_json_schema(),
                options={"temperature": 0},
            )
            parsed = TripletExtraction.model_validate(extraction)
            if parsed.triplets:
                LOGGER.debug("Extracted %s triplets via LLM", len(parsed.triplets))
                return parsed.triplets
        except (ResponseError, ValueError, RuntimeError) as exc:
            LOGGER.warning("LLM triplet extraction failed, falling back to rules: %s", exc)
        fallback = self.fallback_triplets(text)
        LOGGER.debug("Extracted %s triplets via fallback rules", len(fallback))
        return fallback

    def store(
        self,
        *,
        memory_id: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        serializable_metadata = {key: str(value) for key, value in metadata.items()}
        self.memory_collection.upsert(
            ids=[memory_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[serializable_metadata],
        )
        LOGGER.info(
            "Stored memory id=%s type=%s embedding_dims=%s",
            memory_id,
            serializable_metadata.get("memory_type", "unknown"),
            len(embedding),
        )

    def ingest(self, *, source: str, content: str) -> dict[str, Any]:
        document_id = self.db.insert_document(source=source, content=content)
        chunks = self.chunk_text(content)
        for index, chunk in enumerate(chunks):
            embedding = self.safe_embed(chunk)
            self.db.insert_chunk(
                document_id=document_id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
                metadata={"source": source},
            )
            self.store(
                memory_id=f"document-{document_id}-chunk-{index}",
                text=chunk,
                embedding=embedding,
                metadata={
                    "memory_type": "document_chunk",
                    "document_id": document_id,
                    "chunk_index": index,
                    "source": source,
                },
            )

        triplets = self.extract_triplets(content)
        self.db.insert_triplets(document_id, triplets)
        LOGGER.info(
            "Ingested source=%s document_id=%s chunks=%s triplets=%s",
            source,
            document_id,
            len(chunks),
            len(triplets),
        )
        return {
            "document_id": document_id,
            "chunks_indexed": len(chunks),
            "triplets_indexed": len(triplets),
        }

    def retrieve(self, *, session_id: str, query: str) -> RetrievalBundle:
        query_embedding = self.safe_embed(query)
        results = self.memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        top_chunks: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            score = 0.0 if distance is None else 1.0 / (1.0 + float(distance))
            top_chunks.append(
                {
                    "chunk_id": metadata.get("chunk_index", metadata.get("memory_id", f"memory-{index}")),
                    "document_id": metadata.get("document_id"),
                    "chunk_index": metadata.get("chunk_index"),
                    "content": document,
                    "metadata": metadata,
                    "score": score,
                }
            )

        graph_rows = self.db.fetch_triplets_for_terms(self.normalize_terms(query), limit=max(20, self.top_k * 4))
        triplets = [dict(row) for row in graph_rows]
        summary = self.db.fetch_summary(session_id)
        LOGGER.info(
            "Retrieved context for session=%s query=%r chunks=%s triplets=%s",
            session_id,
            query,
            len(top_chunks),
            len(triplets),
        )
        return RetrievalBundle(summary=summary, chunks=top_chunks, triplets=triplets)

    def remember_turn(self, session_id: str, role: str, content: str) -> None:
        self.db.append_turn(session_id, role, content)

    def store_interaction_memory(self, *, session_id: str, query: str, answer: str, run_id: str) -> dict[str, int]:
        query_embedding = self.safe_embed(query)
        answer_embedding = self.safe_embed(answer)
        self.store(
            memory_id=f"{run_id}-query",
            text=query,
            embedding=query_embedding,
            metadata={"memory_type": "user_query", "session_id": session_id, "run_id": run_id},
        )
        self.store(
            memory_id=f"{run_id}-answer",
            text=answer,
            embedding=answer_embedding,
            metadata={"memory_type": "assistant_answer", "session_id": session_id, "run_id": run_id},
        )
        triplets = self.extract_triplets(f"User query: {query}\nAssistant answer: {answer}")
        memory_document_id = self.db.insert_document(
            source=f"interaction:{run_id}",
            content=f"Query: {query}\nAnswer: {answer}",
        )
        self.db.insert_triplets(memory_document_id, triplets)
        LOGGER.info(
            "Stored interaction memory run_id=%s session_id=%s query_dims=%s answer_dims=%s triplets=%s",
            run_id,
            session_id,
            len(query_embedding),
            len(answer_embedding),
            len(triplets),
        )
        return {
            "query_embedding_dimensions": len(query_embedding),
            "answer_embedding_dimensions": len(answer_embedding),
            "triplets_indexed": len(triplets),
        }

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
        LOGGER.debug("Updated mid-term summary for session=%s", session_id)
        return {"summary_updated": True, "window_turns": [dict(row) for row in turns[-rolling_window:]]}
