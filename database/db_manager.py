from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class DatabaseManager:
    """Owns the persistent SQLite memory bank, workflow state, and knowledge graph schema."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS triplets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                );

                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL UNIQUE,
                    summary TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    query TEXT,
                    status TEXT NOT NULL,
                    active_agent TEXT,
                    started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT,
                    final_answer TEXT,
                    judge_approved INTEGER DEFAULT 0,
                    attempts INTEGER DEFAULT 0,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT DEFAULT '',
                    payload TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES workflow_runs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_triplets_document_id ON triplets(document_id);
                CREATE INDEX IF NOT EXISTS idx_triplets_subject ON triplets(subject);
                CREATE INDEX IF NOT EXISTS idx_triplets_object ON triplets(object);
                CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_turns(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_session ON workflow_runs(session_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, id);
                """
            )

    def insert_document(self, source: str, content: str) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO documents (source, content) VALUES (?, ?)",
                (source, content),
            )
            return int(cursor.lastrowid)

    def insert_chunk(
        self,
        *,
        document_id: int,
        chunk_index: int,
        content: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (document_id, chunk_index, content, json.dumps(embedding), json.dumps(metadata)),
            )

    def insert_triplets(self, document_id: int, triplets: list[dict]) -> None:
        if not triplets:
            return
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO triplets (document_id, subject, predicate, object, weight)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        item["subject"],
                        item["predicate"],
                        item["object"],
                        float(item.get("weight", 1.0)),
                    )
                    for item in triplets
                ],
            )

    def fetch_chunks(self) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT id, document_id, chunk_index, content, embedding, metadata FROM chunks"
            ).fetchall()

    def fetch_triplets_for_terms(self, terms: list[str], limit: int = 20) -> list[sqlite3.Row]:
        if not terms:
            return []
        placeholders = ",".join("?" for _ in terms)
        with self.connection() as conn:
            return conn.execute(
                f"""
                SELECT document_id, subject, predicate, object, weight
                FROM triplets
                WHERE lower(subject) IN ({placeholders}) OR lower(object) IN ({placeholders})
                ORDER BY weight DESC, id DESC
                LIMIT ?
                """,
                [*terms, *terms, limit],
            ).fetchall()

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO conversation_turns (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def fetch_turns(self, session_id: str) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT role, content FROM conversation_turns WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

    def upsert_summary(self, session_id: str, summary: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO summaries (session_id, summary, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, summary),
            )

    def fetch_summary(self, session_id: str) -> str | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT summary FROM summaries WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return None if row is None else str(row[0])

    def create_workflow_run(self, run_id: str, session_id: str, query: str | None) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO workflow_runs (id, session_id, query, status, active_agent, attempts)
                VALUES (?, ?, ?, 'queued', 'conductor', 0)
                """,
                (run_id, session_id, query),
            )

    def update_workflow_run(
        self,
        run_id: str,
        *,
        status: str,
        active_agent: str | None = None,
        final_answer: str | None = None,
        judge_approved: bool | None = None,
        attempts: int | None = None,
        error_message: str | None = None,
        completed: bool = False,
    ) -> None:
        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list[object] = [status]
        if active_agent is not None:
            updates.append("active_agent = ?")
            params.append(active_agent)
        if final_answer is not None:
            updates.append("final_answer = ?")
            params.append(final_answer)
        if judge_approved is not None:
            updates.append("judge_approved = ?")
            params.append(1 if judge_approved else 0)
        if attempts is not None:
            updates.append("attempts = ?")
            params.append(attempts)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if completed:
            updates.append("completed_at = CURRENT_TIMESTAMP")
        params.append(run_id)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE workflow_runs SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    def append_run_event(
        self,
        *,
        run_id: str,
        agent_name: str,
        stage: str,
        status: str,
        detail: str,
        payload: dict | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO run_events (run_id, agent_name, stage, status, detail, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, agent_name, stage, status, detail, json.dumps(payload or {})),
            )

    def fetch_workflow_run(self, run_id: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM workflow_runs WHERE id = ?", (run_id,)).fetchone()

    def fetch_run_events(self, run_id: str) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT agent_name, stage, status, detail, payload, created_at FROM run_events WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()

    def list_workflow_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM workflow_runs ORDER BY updated_at DESC, started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

    def fetch_metrics(self) -> dict[str, int]:
        with self.connection() as conn:
            docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            triplets = conn.execute("SELECT COUNT(*) FROM triplets").fetchone()[0]
            sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM conversation_turns").fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM workflow_runs").fetchone()[0]
        return {
            "documents": int(docs),
            "chunks": int(chunks),
            "triplets": int(triplets),
            "sessions": int(sessions),
            "workflow_runs": int(runs),
        }
