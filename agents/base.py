from __future__ import annotations

import json
import os
from abc import ABC
from typing import Any

from ollama import Client


class BaseAgent(ABC):
    """Shared Ollama client and JSON-generation helpers for all agents."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        host: str,
        keep_alive: str = "20m",
        headers: dict[str, str] | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.name = name
        self.model = model
        self.keep_alive = keep_alive
        self.client = Client(host=host, headers=headers or {}, timeout=timeout)

    @staticmethod
    def _schema_payload(schema: dict[str, Any] | None) -> dict[str, Any] | str:
        return schema or "json"

    def chat_json(
        self,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=self._schema_payload(schema),
            options=options or {},
            keep_alive=self.keep_alive,
        )
        content = response["message"]["content"]
        if isinstance(content, dict):
            return content
        return json.loads(content)

    def embed(self, text: str, *, model: str) -> list[float]:
        response = self.client.embed(model=model, input=text, keep_alive=self.keep_alive)
        embeddings = response.get("embeddings") or []
        if not embeddings:
            raise ValueError(f"{self.name} received no embeddings from model {model}.")
        return embeddings[0]

    def ping(self) -> dict[str, Any]:
        return self.client.list()

    @staticmethod
    def headers_from_env() -> dict[str, str]:
        raw = os.getenv("OLLAMA_HEADERS", "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("OLLAMA_HEADERS must be valid JSON.") from exc
