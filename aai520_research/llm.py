"""Language-model interface.

``build_llm`` selects a hosted OpenAI-compatible model when an API key is
present and a deterministic heuristic provider otherwise. Both providers
expose ``complete_json`` so the agent code does not branch on the provider.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when a hosted model request or response cannot be used."""


class LLM(Protocol):
    """Minimal chat interface used by every workflow."""

    name: str

    def complete_json(self, system: str, user: str) -> dict:
        """Return one JSON object for a system prompt and user payload."""


def task_id(system: str) -> str:
    """Read the ``[TASK:name]`` tag that selects a heuristic handler."""
    match = re.search(r"\[TASK:([a-z0-9_]+)\]", system)
    if not match:
        raise LLMError("Prompt is missing a [TASK:...] tag.")
    return match.group(1)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object, including responses wrapped in a code fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise LLMError("The model did not return a JSON object.") from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise LLMError("The model JSON was not an object.")
    return data


class OpenAICompatibleLLM:
    """Chat client for the OpenAI API and compatible base URLs."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = f"openai:{model}"

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        return cls(api_key=api_key, model=model, base_url=base_url)

    def complete_json(self, system: str, user: str) -> dict:
        try:
            text = self._request(system, user, force_json=True)
        except LLMError as exc:
            if "response_format" not in str(exc):
                raise
            text = self._request(system, user, force_json=False)
        return parse_json_object(text)

    def _request(self, system: str, user: str, force_json: bool) -> str:
        body: dict = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if force_json:
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"LLM request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM request failed: {exc.reason}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response did not include message content.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM response content was empty.")
        return content


def build_llm(mode: str | None = None) -> LLM:
    """Select the heuristic provider or a hosted model.

    ``LLM_MODE`` may be ``auto`` (default), ``heuristic``, or ``openai``.
    ``auto`` uses a hosted model only when ``OPENAI_API_KEY`` is set.
    """
    selected = (mode or os.getenv("LLM_MODE") or "auto").strip().lower()
    if selected == "heuristic":
        from aai520_research.heuristic import HeuristicLLM

        return HeuristicLLM()
    if selected == "openai":
        return OpenAICompatibleLLM.from_env()
    if selected != "auto":
        raise LLMError("LLM_MODE must be auto, heuristic, or openai.")
    if os.getenv("OPENAI_API_KEY", "").strip():
        return OpenAICompatibleLLM.from_env()
    from aai520_research.heuristic import HeuristicLLM

    return HeuristicLLM()
