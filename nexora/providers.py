"""Provider-independent streaming chat over Gemini, Claude, OpenAI and Ollama.

Each provider turns (system prompt, history) into a stream of text chunks and fills
in a Usage object when the provider reports token counts. Errors are translated into
ProviderError with a message that is safe to show users; the original exception is
kept as __cause__ for server logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from nexora.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderError(RuntimeError):
    """A provider call failed. `user_message` is safe to show; `retryable` drives retries."""

    def __init__(self, user_message: str, retryable: bool = False):
        super().__init__(user_message)
        self.user_message = user_message
        self.retryable = retryable


class ChatProvider(Protocol):
    name: str
    model: str

    def stream(self, system: str, messages: list[ChatMessage], usage: Usage) -> AsyncIterator[str]: ...


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def classify(exc: BaseException, provider: str) -> ProviderError:
    """Map an SDK/HTTP exception to a ProviderError without leaking details to users."""
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)) or "timeout" in type(exc).__name__.lower():
        return ProviderError(f"{provider} took too long to respond. Please try again.", retryable=True)
    if isinstance(exc, httpx.ConnectError) or "connection" in type(exc).__name__.lower():
        return ProviderError(f"Couldn't reach {provider}. Check the network or service URL.", retryable=True)
    status = _status_of(exc)
    if status in (401, 403):
        return ProviderError(f"{provider} rejected the API key or the key lacks access to this model.")
    if status == 404:
        return ProviderError(f"{provider} doesn't recognise the configured model name.")
    if status == 429:
        return ProviderError(f"{provider} rate limit or quota reached. Wait a moment and try again.", retryable=True)
    if status is not None and status >= 500:
        return ProviderError(f"{provider} had a server error. Please try again.", retryable=True)
    if status == 400:
        return ProviderError(f"{provider} rejected the request (it may be too long or blocked by safety filters).")
    return ProviderError(f"Unexpected error from {provider}.")


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: int, max_output_tokens: int):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
        self.model = model
        self._max_output_tokens = max_output_tokens

    async def stream(self, system: str, messages: list[ChatMessage], usage: Usage) -> AsyncIterator[str]:
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]} for m in messages
        ]
        config = self._types.GenerateContentConfig(system_instruction=system, max_output_tokens=self._max_output_tokens)
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )
        async for chunk in stream:
            meta = getattr(chunk, "usage_metadata", None)
            if meta is not None:
                usage.input_tokens = meta.prompt_token_count or usage.input_tokens
                usage.output_tokens = meta.candidates_token_count or usage.output_tokens
            if chunk.text:
                yield chunk.text


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout_s: int, max_output_tokens: int):
        from anthropic import AsyncAnthropic

        # Retries are handled in stream_reply so they never repeat partial output.
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self.model = model
        self._max_output_tokens = max_output_tokens

    async def stream(self, system: str, messages: list[ChatMessage], usage: Usage) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=self.model,
            max_tokens=self._max_output_tokens,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
            usage.input_tokens = final.usage.input_tokens
            usage.output_tokens = final.usage.output_tokens


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, timeout_s: int, max_output_tokens: int):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
        self.model = model
        self._max_output_tokens = max_output_tokens

    async def stream(self, system: str, messages: list[ChatMessage], usage: Usage) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            max_completion_tokens=self._max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage.input_tokens = chunk.usage.prompt_tokens
                usage.output_tokens = chunk.usage.completion_tokens
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_s: int, max_output_tokens: int):
        self._base_url = base_url
        self._timeout = timeout_s
        self.model = model
        self._max_output_tokens = max_output_tokens

    async def stream(self, system: str, messages: list[ChatMessage], usage: Usage) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "stream": True,
            "options": {"num_predict": self._max_output_tokens},
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
        }
        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise ProviderError(f"Ollama error: {data['error']}")
                    text = (data.get("message") or {}).get("content")
                    if text:
                        yield text
                    if data.get("done"):
                        usage.input_tokens = data.get("prompt_eval_count")
                        usage.output_tokens = data.get("eval_count")


def build_provider(settings: Settings, name: str, model: str | None = None) -> ChatProvider:
    if name not in settings.providers:
        raise ProviderError(f"Provider {name!r} is not configured on this server.")
    model = (model or "").strip() or settings.models[name]
    args = (model, settings.llm_timeout_s, settings.max_output_tokens)
    if name == "gemini":
        return GeminiProvider(settings.api_keys["gemini"], *args)
    if name == "anthropic":
        return AnthropicProvider(settings.api_keys["anthropic"], *args)
    if name == "openai":
        return OpenAIProvider(settings.api_keys["openai"], *args)
    return OllamaProvider(settings.ollama_base_url, *args)


async def stream_reply(
    provider: ChatProvider,
    system: str,
    messages: list[ChatMessage],
    usage: Usage,
    timeout_s: int,
    max_attempts: int = 3,
) -> AsyncIterator[str]:
    """Stream a reply with an overall timeout. Retries transient failures only
    before the first chunk arrives, so users never see duplicated text."""
    for attempt in range(1, max_attempts + 1):
        produced = False
        try:
            async with asyncio.timeout(timeout_s):
                async for chunk in provider.stream(system, messages, usage):
                    produced = True
                    yield chunk
            return
        except Exception as exc:
            error = classify(exc, provider.name)
            logger.warning(
                "%s/%s attempt %d failed: %s: %s",
                provider.name,
                provider.model,
                attempt,
                type(exc).__name__,
                exc,
            )
            if produced or not error.retryable or attempt == max_attempts:
                raise error from exc
            await asyncio.sleep(2 ** (attempt - 1))
