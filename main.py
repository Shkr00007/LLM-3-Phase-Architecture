from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

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
        # structural assertion for strict handoff format
        required = STRICT_ENVELOPE_SCHEMA["required"]
        if any(key not in packet for key in required):
            raise ValueError("Envelope missing required keys.")
        return packet

    def check_connection(self) -> None:
        self.repository.ping()
        print("Connected to Mobius Sales AI server.")

    def ingest_file(self, path: str) -> dict[str, Any]:
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        result = self.repository.ingest(source=str(file_path), content=content)
        return self.envelope("conductor", "repository", "ingest_result", result)

    def run_query(self, *, session_id: str, query: str) -> dict[str, Any]:
        self.repository.remember_turn(session_id, "user", query)
        feedback: str | None = None
        for attempt in range(self.max_retries + 1):
            diplomat_packet = self.diplomat.plan_and_draft(
                session_id=session_id,
                user_query=query,
                repository=self.repository,
                judge_feedback=feedback,
            )
            judge_packet = self.judge.audit(user_query=query, draft_packet=diplomat_packet)
            audited = self.envelope(
                "judge",
                "conductor",
                "audit_result",
                {
                    "attempt": attempt + 1,
                    "draft_packet": diplomat_packet,
                    "judge_verdict": judge_packet,
                },
            )
            if judge_packet["approved"]:
                final_answer = diplomat_packet["draft"]["answer"]
                self.repository.remember_turn(session_id, "assistant", final_answer)
                return audited
            feedback = judge_packet["retry_feedback"]
        final_answer = diplomat_packet["draft"]["answer"]
        self.repository.remember_turn(session_id, "assistant", final_answer)
        return audited


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3-phase multi-agent conductor for remote Ollama.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Verify remote Ollama connectivity.")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest a text file into persistent memory.")
    ingest_parser.add_argument("path")

    query_parser = subparsers.add_parser("query", help="Run the multi-agent query pipeline.")
    query_parser.add_argument("query")
    query_parser.add_argument("--session-id", default=f"session-{uuid4()}")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    conductor = Conductor()

    if args.command == "check":
        conductor.check_connection()
        return
    if args.command == "ingest":
        print(json.dumps(conductor.ingest_file(args.path), indent=2))
        return
    if args.command == "query":
        result = conductor.run_query(session_id=args.session_id, query=args.query)
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
