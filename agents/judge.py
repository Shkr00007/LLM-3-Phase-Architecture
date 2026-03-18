from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.base import BaseAgent


class JudgeVerdict(BaseModel):
    approved: bool
    score: float = Field(ge=0, le=1)
    issues: list[str] = Field(default_factory=list)
    retry_feedback: str = ""


class JudgeAgent(BaseAgent):
    def __init__(self, *, host: str, model: str, keep_alive: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(
            name="judge",
            model=model,
            host=host,
            keep_alive=keep_alive,
            headers=headers,
        )

    def audit(self, *, user_query: str, draft_packet: dict[str, Any]) -> dict[str, Any]:
        result = self.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Judge Agent performing LLM-as-a-Judge verification. "
                        "Approve only if the answer is supported by the provided chunks, triplets, or summary. "
                        "Flag missing support, contradictions, or speculative claims, and provide targeted retry guidance."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"User query: {user_query}\n"
                        f"Draft packet to audit: {draft_packet}"
                    ),
                },
            ],
            schema=JudgeVerdict.model_json_schema(),
            options={"temperature": 0},
        )
        verdict = JudgeVerdict.model_validate(result)
        return verdict.model_dump()
