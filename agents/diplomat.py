from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agents.base import BaseAgent
from agents.repository import RetrievalBundle, RepositoryAgent


LOGGER = logging.getLogger(__name__)


class ReActStep(BaseModel):
    thought: str
    action: str
    observation: str


class DiplomatDraft(BaseModel):
    answer: str
    grounded: bool = True
    confidence: float = Field(ge=0, le=1)
    citations: list[str] = Field(default_factory=list)
    react_trace: list[ReActStep] = Field(default_factory=list)
    follow_up: list[str] = Field(default_factory=list)


class DiplomatAgent(BaseAgent):
    def __init__(
        self,
        *,
        host: str,
        model: str,
        keep_alive: str,
        headers: dict[str, str] | None = None,
        rolling_window: int = 6,
    ) -> None:
        super().__init__(
            name="diplomat",
            model=model,
            host=host,
            keep_alive=keep_alive,
            headers=headers,
        )
        self.rolling_window = rolling_window

    @staticmethod
    def _context_text(bundle: RetrievalBundle, recent_turns: list[dict[str, str]]) -> str:
        chunk_text = "\n\n".join(
            f"[Chunk {item['chunk_id']} | score={item['score']:.3f}] {item['content']}"
            for item in bundle.chunks
        )
        triplet_text = "\n".join(
            f"({item['subject']}, {item['predicate']}, {item['object']}) weight={item['weight']}"
            for item in bundle.triplets
        )
        conversation_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent_turns)
        return (
            f"Mid-term memory summary:\n{bundle.summary or 'None'}\n\n"
            f"Recent rolling window:\n{conversation_text or 'None'}\n\n"
            f"Top retrieved chunks:\n{chunk_text or 'None'}\n\n"
            f"Knowledge graph triplets:\n{triplet_text or 'None'}"
        )

    def plan_and_draft(
        self,
        *,
        session_id: str,
        user_query: str,
        repository: RepositoryAgent,
        judge_feedback: str | None = None,
    ) -> dict[str, Any]:
        prune_result = repository.prune_context(session_id=session_id, rolling_window=self.rolling_window)
        bundle = repository.retrieve(session_id=session_id, query=user_query)
        prompt = self._context_text(bundle, prune_result["window_turns"])
        feedback_block = f"Judge feedback to address on retry: {judge_feedback}" if judge_feedback else ""
        has_context = bool(bundle.chunks or bundle.triplets or bundle.summary)

        if has_context:
            system_prompt = (
                "You are the Diplomat Agent. Use a concise ReAct style internally, ground every claim in retrieved memory, "
                "and do not invent facts not present in context. Set grounded=true. Return only JSON matching the schema."
            )
        else:
            LOGGER.info("Fallback LLM answer used (no prior context) for session=%s", session_id)
            system_prompt = (
                "You are the Diplomat Agent. No prior repository context was retrieved. Answer the user directly using your "
                "general model knowledge, mark grounded=false, lower confidence accordingly, and clearly avoid pretending the "
                "answer came from memory. Return only JSON matching the schema."
            )

        response = self.chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Session: {session_id}\nUser query: {user_query}\n\n{feedback_block}\n\n{prompt}"},
            ],
            schema=DiplomatDraft.model_json_schema(),
            options={"temperature": 0.2},
        )
        parsed = DiplomatDraft.model_validate(response)
        return {
            "session_id": session_id,
            "query": user_query,
            "context": {
                "summary": bundle.summary,
                "chunks": bundle.chunks,
                "triplets": bundle.triplets,
            },
            "draft": parsed.model_dump(),
            "has_context": has_context,
        }
