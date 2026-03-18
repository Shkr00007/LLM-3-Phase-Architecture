from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.base import BaseAgent
from agents.diplomat import DiplomatAgent
from agents.judge import JudgeAgent
from agents.repository import RepositoryAgent
from database.db_manager import DatabaseManager


STRICT_ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "sender": {"type": "string"},
        "recipient": {"type": "string"},
        "message_type": {"type": "string"},
        "payload": {"type": "object"},
    },
    "required": ["sender", "recipient", "message_type", "payload"],
    "additionalProperties": False,
}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None


class IngestRequest(BaseModel):
    path: str


class RunEvent(BaseModel):
    agent_name: str
    stage: str
    status: str
    detail: str
    payload: dict[str, Any]
    created_at: str


class RunStatusResponse(BaseModel):
    run_id: str
    session_id: str
    query: str | None
    status: str
    active_agent: str | None
    attempts: int
    judge_approved: bool
    started_at: str
    updated_at: str
    completed_at: str | None
    final_answer: str | None
    error_message: str | None
    events: list[RunEvent]


class Conductor:
    def __init__(self) -> None:
        load_dotenv()
        host = os.getenv("OLLAMA_HOST", "https://ollama-mobius-sales.mobiusdtaas.ai")
        headers = BaseAgent.headers_from_env()
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "20m")
        db_path = os.getenv("MEMORY_DB_PATH", "database/memory.db")
        top_k = int(os.getenv("REPOSITORY_TOP_K", "5"))
        window_size = int(os.getenv("DIPLOMAT_WINDOW_SIZE", "6"))
        self.max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "2"))

        self.db = DatabaseManager(db_path)
        self.repository = RepositoryAgent(
            host=host,
            model=os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:14b"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            db=self.db,
            keep_alive=keep_alive,
            headers=headers,
            top_k=top_k,
        )
        self.diplomat = DiplomatAgent(
            host=host,
            model=os.getenv("OLLAMA_INTERACTION_MODEL", "qwen3:7b"),
            keep_alive=keep_alive,
            headers=headers,
            rolling_window=window_size,
        )
        self.judge = JudgeAgent(
            host=host,
            model=os.getenv("OLLAMA_REASONING_MODEL", "deepseek-r1:14b"),
            keep_alive=keep_alive,
            headers=headers,
        )

    @staticmethod
    def envelope(sender: str, recipient: str, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        packet = {
            "sender": sender,
            "recipient": recipient,
            "message_type": message_type,
            "payload": payload,
        }
        required = STRICT_ENVELOPE_SCHEMA["required"]
        if any(key not in packet for key in required):
            raise ValueError("Envelope missing required keys.")
        return packet

    def log_event(
        self,
        *,
        run_id: str,
        agent_name: str,
        stage: str,
        status: str,
        detail: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.db.append_run_event(
            run_id=run_id,
            agent_name=agent_name,
            stage=stage,
            status=status,
            detail=detail,
            payload=payload,
        )

    def check_connection(self) -> dict[str, Any]:
        inventory = self.repository.ping()
        return self.envelope(
            "repository",
            "conductor",
            "connection_status",
            {"connected": True, "models": inventory.get("models", [])},
        )

    def ingest_file(self, path: str) -> dict[str, Any]:
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        result = self.repository.ingest(source=str(file_path), content=content)
        return self.envelope("conductor", "repository", "ingest_result", result)

    def run_query(self, *, session_id: str, query: str, run_id: str | None = None) -> dict[str, Any]:
        run_id = run_id or f"run-{uuid4()}"
        self.db.create_workflow_run(run_id=run_id, session_id=session_id, query=query)
        self.db.update_workflow_run(run_id, status="running", active_agent="repository")
        self.log_event(
            run_id=run_id,
            agent_name="conductor",
            stage="query_received",
            status="completed",
            detail="Conductor accepted a new query.",
            payload={"session_id": session_id, "query": query},
        )
        self.repository.remember_turn(session_id, "user", query)

        feedback: str | None = None
        diplomat_packet: dict[str, Any] | None = None
        judge_packet: dict[str, Any] | None = None
        try:
            self.log_event(
                run_id=run_id,
                agent_name="repository",
                stage="memory_update",
                status="completed",
                detail="Stored the incoming user turn in persistent memory.",
                payload={"session_id": session_id},
            )
            for attempt in range(self.max_retries + 1):
                self.db.update_workflow_run(run_id, status="running", active_agent="diplomat", attempts=attempt + 1)
                self.log_event(
                    run_id=run_id,
                    agent_name="diplomat",
                    stage="drafting",
                    status="running",
                    detail=f"Draft attempt {attempt + 1} started.",
                    payload={"judge_feedback": feedback or ""},
                )
                diplomat_packet = self.diplomat.plan_and_draft(
                    session_id=session_id,
                    user_query=query,
                    repository=self.repository,
                    judge_feedback=feedback,
                )
                self.log_event(
                    run_id=run_id,
                    agent_name="diplomat",
                    stage="drafting",
                    status="completed",
                    detail=f"Draft attempt {attempt + 1} completed.",
                    payload={
                        "confidence": diplomat_packet["draft"]["confidence"],
                        "citations": diplomat_packet["draft"]["citations"],
                    },
                )

                self.db.update_workflow_run(run_id, status="running", active_agent="judge", attempts=attempt + 1)
                self.log_event(
                    run_id=run_id,
                    agent_name="judge",
                    stage="audit",
                    status="running",
                    detail=f"Judge audit attempt {attempt + 1} started.",
                    payload={},
                )
                judge_packet = self.judge.audit(user_query=query, draft_packet=diplomat_packet)
                self.log_event(
                    run_id=run_id,
                    agent_name="judge",
                    stage="audit",
                    status="completed",
                    detail=f"Judge audit attempt {attempt + 1} finished.",
                    payload=judge_packet,
                )

                audited = self.envelope(
                    "judge",
                    "conductor",
                    "audit_result",
                    {
                        "run_id": run_id,
                        "attempt": attempt + 1,
                        "draft_packet": diplomat_packet,
                        "judge_verdict": judge_packet,
                    },
                )
                final_answer = diplomat_packet["draft"]["answer"]
                if judge_packet["approved"]:
                    self.repository.remember_turn(session_id, "assistant", final_answer)
                    self.db.update_workflow_run(
                        run_id,
                        status="completed",
                        active_agent="conductor",
                        final_answer=final_answer,
                        judge_approved=True,
                        attempts=attempt + 1,
                        completed=True,
                    )
                    self.log_event(
                        run_id=run_id,
                        agent_name="conductor",
                        stage="finalize",
                        status="completed",
                        detail="Query approved and response stored.",
                        payload={"answer_preview": final_answer[:500]},
                    )
                    return audited
                feedback = judge_packet["retry_feedback"]
                self.log_event(
                    run_id=run_id,
                    agent_name="conductor",
                    stage="retry",
                    status="running",
                    detail=f"Retry requested after attempt {attempt + 1}.",
                    payload={"retry_feedback": feedback},
                )

            final_answer = diplomat_packet["draft"]["answer"] if diplomat_packet else None
            if final_answer:
                self.repository.remember_turn(session_id, "assistant", final_answer)
            self.db.update_workflow_run(
                run_id,
                status="completed_with_warnings",
                active_agent="conductor",
                final_answer=final_answer,
                judge_approved=False,
                attempts=self.max_retries + 1,
                error_message="Judge never approved the draft before retry budget was exhausted.",
                completed=True,
            )
            self.log_event(
                run_id=run_id,
                agent_name="conductor",
                stage="finalize",
                status="warning",
                detail="Returned the last draft after exhausting retries.",
                payload={"judge_verdict": judge_packet or {}},
            )
            return audited
        except Exception as exc:
            self.db.update_workflow_run(
                run_id,
                status="failed",
                active_agent="conductor",
                error_message=str(exc),
                completed=True,
            )
            self.log_event(
                run_id=run_id,
                agent_name="conductor",
                stage="failure",
                status="failed",
                detail="Workflow failed with an exception.",
                payload={"error": str(exc)},
            )
            raise

    def get_run_status(self, run_id: str) -> RunStatusResponse:
        row = self.db.fetch_workflow_run(run_id)
        if row is None:
            raise KeyError(run_id)
        events = []
        for event in self.db.fetch_run_events(run_id):
            payload = json.loads(event["payload"])
            events.append(
                RunEvent(
                    agent_name=event["agent_name"],
                    stage=event["stage"],
                    status=event["status"],
                    detail=event["detail"],
                    payload=payload,
                    created_at=event["created_at"],
                )
            )
        return RunStatusResponse(
            run_id=row["id"],
            session_id=row["session_id"],
            query=row["query"],
            status=row["status"],
            active_agent=row["active_agent"],
            attempts=int(row["attempts"]),
            judge_approved=bool(row["judge_approved"]),
            started_at=row["started_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            final_answer=row["final_answer"],
            error_message=row["error_message"],
            events=events,
        )

    def system_status(self) -> dict[str, Any]:
        runs = [dict(row) for row in self.db.list_workflow_runs(limit=10)]
        return {
            "service": "llm-3-phase-architecture",
            "ollama_host": os.getenv("OLLAMA_HOST", "https://ollama-mobius-sales.mobiusdtaas.ai"),
            "models": {
                "interaction": self.diplomat.model,
                "reasoning": self.judge.model,
                "embedding": self.repository.embed_model,
            },
            "keep_alive": self.repository.keep_alive,
            "metrics": self.db.fetch_metrics(),
            "recent_runs": runs,
        }


conductor = Conductor()
app = FastAPI(title="LLM 3-Phase Architecture", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/check")
def api_check() -> dict[str, Any]:
    return conductor.check_connection()


@app.post("/ingest")
def api_ingest(request: IngestRequest) -> dict[str, Any]:
    return conductor.ingest_file(request.path)


@app.post("/query")
def api_query(request: QueryRequest) -> dict[str, Any]:
    session_id = request.session_id or f"session-{uuid4()}"
    return conductor.run_query(session_id=session_id, query=request.query)


@app.get("/status")
def api_status() -> dict[str, Any]:
    return conductor.system_status()


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
def api_run_status(run_id: str) -> RunStatusResponse:
    try:
        return conductor.get_run_status(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3-phase multi-agent conductor for remote Ollama.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Verify remote Ollama connectivity.")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a text file into persistent memory.")
    ingest_parser.add_argument("path")

    query_parser = subparsers.add_parser("query", help="Run the multi-agent query pipeline.")
    query_parser.add_argument("query")
    query_parser.add_argument("--session-id", default=f"session-{uuid4()}")

    run_parser = subparsers.add_parser("run-status", help="Inspect a workflow run and agent events.")
    run_parser.add_argument("run_id")

    subparsers.add_parser("status", help="Show service-wide metrics and recent runs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        print(json.dumps(conductor.check_connection(), indent=2))
        return
    if args.command == "ingest":
        print(json.dumps(conductor.ingest_file(args.path), indent=2))
        return
    if args.command == "query":
        result = conductor.run_query(session_id=args.session_id, query=args.query)
        print(json.dumps(result, indent=2))
        return
    if args.command == "status":
        print(json.dumps(conductor.system_status(), indent=2))
        return
    if args.command == "run-status":
        print(conductor.get_run_status(args.run_id).model_dump_json(indent=2))
        return


if __name__ == "__main__":
    main()
