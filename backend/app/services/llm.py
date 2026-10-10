"""LLM providers: the generation half of RAG.

Deliberately separate from `generator.py`. This module knows how to talk to a
model; it knows nothing about codebases, retrieval or prompts. That split is
what lets the whole generation pipeline be tested offline against
FakeLLMProvider, and what makes swapping providers an .env change.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from app.config import Settings, get_settings

# USD per 1M tokens (input, output).
LLM_PRICES_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


class LLMError(RuntimeError):
    """Raised when generation fails."""


@dataclass
class LLMResponse:
    """One completion, plus what it cost."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        """True when the model ran out of output budget mid-sentence.

        Worth surfacing: a truncated answer often loses its citations, which
        are usually written last.
        """
        return self.finish_reason == "length"

    @property
    def usd(self) -> float | None:
        price = LLM_PRICES_PER_1M_TOKENS.get(self.model)
        if price is None:
            return None
        input_price, output_price = price
        return round(
            self.input_tokens / 1_000_000 * input_price
            + self.output_tokens / 1_000_000 * output_price,
            6,
        )


class LLMProvider(ABC):
    """Anything that can turn a prompt into text."""

    model: str

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a completion.

        `messages` is the usual [{"role": "user"|"assistant", "content": ...}]
        list, oldest first. The system prompt is passed separately because
        providers place it differently.
        """


# ---------------------------------------------------------------------------
# Fake provider: offline, scripted, free
# ---------------------------------------------------------------------------

class FakeLLMProvider(LLMProvider):
    """Returns a scripted answer and records what it was asked.

    Tests use the recorded prompts to assert that the CONTEXT was assembled
    correctly — that retrieved code actually reached the model, that history
    was included, that the grounding instruction is present. Those are the
    parts we control. The model's own reasoning is not ours to test.
    """

    def __init__(
        self,
        response: str | Callable[[str, list[dict[str, str]]], str] = "A grounded answer [1].",
        model: str = "fake-llm",
        finish_reason: str = "stop",
    ) -> None:
        self.model = model
        self._response = response
        self._finish_reason = finish_reason
        self.calls: list[dict] = []

    @property
    def last_prompt(self) -> str:
        """System prompt plus every message, flattened. Convenient for asserts."""
        if not self.calls:
            return ""
        call = self.calls[-1]
        parts = [call["system"]] + [m["content"] for m in call["messages"]]
        return "\n".join(parts)

    def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system_prompt,
                "messages": [dict(m) for m in messages],
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )

        text = (
            self._response(system_prompt, messages)
            if callable(self._response)
            else self._response
        )
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=sum(len(m["content"]) for m in messages) // 4,
            output_tokens=len(text) // 4,
            finish_reason=self._finish_reason,
        )


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAILLMProvider(LLMProvider):
    """Chat completions via the OpenAI API, or any server that speaks it.

    `base_url` points it at an OpenAI-compatible server such as Ollama, which
    is how the "ollama" provider runs a local model.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 4,
        base_url: str | None = None,
    ) -> None:
        if not api_key.strip() and base_url is None:
            raise LLMError(
                "OPENAI_API_KEY is empty. Add it to .env, or set LLM_PROVIDER=fake "
                "to run offline."
            )

        from openai import OpenAI  # lazy, so offline tests need no SDK

        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key or "unused", base_url=base_url, max_retries=2
        )

    def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
    ) -> LLMResponse:
        from openai import APIConnectionError, APIError, RateLimitError

        payload = [{"role": "system", "content": system_prompt}, *messages]

        for attempt in range(self.max_retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                choice = completion.choices[0]
                usage = completion.usage
                return LLMResponse(
                    text=(choice.message.content or "").strip(),
                    model=completion.model or self.model,
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    finish_reason=choice.finish_reason or "stop",
                )

            except RateLimitError as exc:
                # Same distinction as the embedding provider: "slow down" is
                # transient, "no credits" is permanent.
                from app.services.embeddings import _is_quota_error

                if _is_quota_error(exc):
                    raise LLMError(
                        "OpenAI rejected the request: the account has no credits "
                        "remaining. Add credits at "
                        "https://platform.openai.com/settings/organization/billing/ "
                        "or set LLM_PROVIDER=fake in .env to work offline."
                    ) from exc
                if attempt == self.max_retries - 1:
                    raise
                _sleep_backoff(attempt)

            except APIConnectionError as exc:
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"Could not reach {self._client.base_url}. If this is "
                        "Ollama, start it (ollama serve) and pull the model "
                        f"(ollama pull {self.model})."
                    ) from exc
                _sleep_backoff(attempt)

            except APIError as exc:
                status = getattr(exc, "status_code", None)
                if status is not None and status < 500:
                    raise
                if attempt == self.max_retries - 1:
                    raise
                _sleep_backoff(attempt)

        raise LLMError("exhausted retries")  # pragma: no cover


def _sleep_backoff(attempt: int) -> None:
    delay = (2 ** attempt) * (0.5 + random.random())
    time.sleep(min(delay, 30.0))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    name = settings.llm_provider.strip().lower()

    if name == "fake":
        return FakeLLMProvider()

    if name == "openai":
        return OpenAILLMProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            max_retries=settings.llm_max_retries,
        )

    if name == "ollama":
        return OpenAILLMProvider(
            api_key="",
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            max_retries=settings.llm_max_retries,
        )

    raise LLMError(
        f"Unknown LLM provider {settings.llm_provider!r}. "
        "Supported: 'openai', 'ollama', 'fake'."
    )
