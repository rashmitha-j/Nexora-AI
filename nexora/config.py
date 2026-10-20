"""Settings read from environment variables. Secrets never leave the server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}

DEFAULT_FALLBACKS = {
    "gemini": "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
}


class ConfigError(RuntimeError):
    """Raised when the environment is missing or has invalid settings."""


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass(frozen=True)
class Settings:
    # Chat providers
    default_provider: str
    api_keys: dict[str, str]
    models: dict[str, str]
    fallback_models: dict[str, list[str]]
    ollama_base_url: str
    llm_timeout_s: int
    max_output_tokens: int
    # Request limits
    rate_limit_per_minute: int
    global_rate_limit_per_minute: int
    max_message_chars: int
    history_turns: int
    # RAG
    database_url: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    max_upload_mb: int
    rag_top_k: int
    rag_min_score: float
    # MetaGPT worker
    worker_url: str
    worker_token: str
    # Auth: "user:pbkdf2_sha256$..." entries separated by commas
    users_spec: str = field(repr=False, default="")

    @property
    def providers(self) -> list[str]:
        """Providers that have the configuration they need."""
        names = [p for p in ("gemini", "anthropic", "openai") if self.api_keys.get(p)]
        if self.ollama_base_url:
            names.append("ollama")
        return names

    @property
    def auth_enabled(self) -> bool:
        return bool(self.users_spec)

    @property
    def rag_enabled(self) -> bool:
        return bool(self.database_url) and bool(self.api_keys.get(self.embedding_provider))

    @property
    def jobs_enabled(self) -> bool:
        return bool(self.worker_url and self.worker_token)


def load_settings() -> Settings:
    api_keys = {
        "gemini": _str("GEMINI_API_KEY"),
        "anthropic": _str("ANTHROPIC_API_KEY"),
        "openai": _str("OPENAI_API_KEY"),
    }
    models = {
        "gemini": _str("GEMINI_MODEL", DEFAULT_MODELS["gemini"]),
        "anthropic": _str("ANTHROPIC_MODEL", DEFAULT_MODELS["anthropic"]),
        "openai": _str("OPENAI_MODEL", DEFAULT_MODELS["openai"]),
        "ollama": _str("OLLAMA_MODEL", DEFAULT_MODELS["ollama"]),
    }
    # Tried in order when the main model hits a quota or is overloaded. Each Gemini model has
    # its own free-tier daily quota, so falling back keeps chat available.
    fallback_models = {}
    for p in DEFAULT_MODELS:
        # Set but empty means "no fallbacks", so read the raw value rather than via _str.
        raw = os.getenv(f"{p.upper()}_FALLBACK_MODELS")
        raw = DEFAULT_FALLBACKS.get(p, "") if raw is None else raw
        fallback_models[p] = [m.strip() for m in raw.split(",") if m.strip()]
    embedding_provider = _str("EMBEDDING_PROVIDER", "gemini").lower()
    if embedding_provider not in ("gemini", "openai"):
        raise ConfigError("EMBEDDING_PROVIDER must be 'gemini' or 'openai'")
    default_embedding_model = "gemini-embedding-001" if embedding_provider == "gemini" else "text-embedding-3-small"

    settings = Settings(
        default_provider=_str("NEXORA_PROVIDER", "gemini").lower(),
        api_keys=api_keys,
        models=models,
        fallback_models=fallback_models,
        ollama_base_url=_str("OLLAMA_BASE_URL").rstrip("/"),
        llm_timeout_s=_int("LLM_TIMEOUT_SECONDS", 60),
        max_output_tokens=_int("MAX_OUTPUT_TOKENS", 2048),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 10),
        global_rate_limit_per_minute=_int("GLOBAL_RATE_LIMIT_PER_MINUTE", 30),
        max_message_chars=_int("MAX_MESSAGE_CHARS", 8000),
        history_turns=_int("HISTORY_TURNS", 10),
        database_url=_str("DATABASE_URL"),
        embedding_provider=embedding_provider,
        embedding_model=_str("EMBEDDING_MODEL", default_embedding_model),
        embedding_dim=_int("EMBEDDING_DIM", 768),
        max_upload_mb=_int("MAX_UPLOAD_MB", 10),
        rag_top_k=_int("RAG_TOP_K", 5),
        rag_min_score=_float("RAG_MIN_SCORE", 0.55),
        worker_url=_str("METAGPT_WORKER_URL").rstrip("/"),
        worker_token=_str("METAGPT_WORKER_TOKEN"),
        users_spec=_str("NEXORA_USERS"),
    )
    validate(settings)
    return settings


def validate(settings: Settings) -> None:
    if not settings.providers:
        raise ConfigError(
            "No chat provider is configured. Set at least one of GEMINI_API_KEY, "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY or OLLAMA_BASE_URL."
        )
    if settings.default_provider not in DEFAULT_MODELS:
        raise ConfigError(f"NEXORA_PROVIDER must be one of {', '.join(DEFAULT_MODELS)}")
    if settings.default_provider not in settings.providers:
        raise ConfigError(f"NEXORA_PROVIDER is {settings.default_provider!r} but that provider is not configured.")
    if settings.database_url and not settings.api_keys.get(settings.embedding_provider):
        raise ConfigError(f"DATABASE_URL is set but EMBEDDING_PROVIDER={settings.embedding_provider!r} has no API key.")
    if bool(settings.worker_url) != bool(settings.worker_token):
        raise ConfigError("Set both METAGPT_WORKER_URL and METAGPT_WORKER_TOKEN, or neither.")
