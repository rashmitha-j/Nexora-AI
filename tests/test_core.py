import asyncio

import pytest

from nexora import config as config_mod
from nexora.auth import authenticate, hash_password, parse_users, verify_password
from nexora.commands import parse_command
from nexora.limits import RateLimiter
from nexora.providers import ChatMessage, ProviderError, Usage, build_provider, classify, stream_reply

# -- config ------------------------------------------------------------------

ENV_KEYS = [
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
    "NEXORA_PROVIDER",
    "DATABASE_URL",
    "EMBEDDING_PROVIDER",
    "METAGPT_WORKER_URL",
    "METAGPT_WORKER_TOKEN",
    "NEXORA_USERS",
    "CHAINLIT_AUTH_SECRET",
    "RATE_LIMIT_PER_MINUTE",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODELS",
]


@pytest.fixture
def env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_config_requires_a_provider(env):
    with pytest.raises(config_mod.ConfigError, match="No chat provider"):
        config_mod.load_settings()


def test_config_defaults_with_gemini(env):
    env.setenv("GEMINI_API_KEY", "k")
    s = config_mod.load_settings()
    assert s.providers == ["gemini"]
    assert s.models["gemini"] == "gemini-3.5-flash"
    assert not s.rag_enabled and not s.jobs_enabled and not s.auth_enabled


def test_config_rejects_unconfigured_default_provider(env):
    env.setenv("GEMINI_API_KEY", "k")
    env.setenv("NEXORA_PROVIDER", "anthropic")
    with pytest.raises(config_mod.ConfigError, match="not configured"):
        config_mod.load_settings()


def test_config_worker_needs_url_and_token(env):
    env.setenv("GEMINI_API_KEY", "k")
    env.setenv("METAGPT_WORKER_URL", "http://w")
    with pytest.raises(config_mod.ConfigError, match="METAGPT_WORKER_TOKEN"):
        config_mod.load_settings()


def test_config_auth_needs_chainlit_secret(env):
    env.setenv("GEMINI_API_KEY", "k")
    env.setenv("NEXORA_USERS", "a:x")
    with pytest.raises(config_mod.ConfigError, match="CHAINLIT_AUTH_SECRET"):
        config_mod.load_settings()


def test_config_bad_integer(env):
    env.setenv("GEMINI_API_KEY", "k")
    env.setenv("RATE_LIMIT_PER_MINUTE", "lots")
    with pytest.raises(config_mod.ConfigError, match="integer"):
        config_mod.load_settings()


def test_build_provider_rejects_unconfigured(env):
    env.setenv("GEMINI_API_KEY", "k")
    s = config_mod.load_settings()
    with pytest.raises(ProviderError):
        build_provider(s, "openai")


def test_build_provider_uses_override_model(env):
    env.setenv("GEMINI_API_KEY", "k")
    s = config_mod.load_settings()
    assert build_provider(s, "gemini", "gemini-3.8-flash").model == "gemini-3.8-flash"
    assert build_provider(s, "gemini", "  ").model == "gemini-3.5-flash"


# -- auth --------------------------------------------------------------------


def test_password_round_trip():
    encoded = hash_password("correct horse", iterations=1000)
    assert verify_password("correct horse", encoded)
    assert not verify_password("wrong", encoded)
    assert not verify_password("x", "garbage")


def test_parse_users_and_authenticate():
    alice = hash_password("pw-alice-123", iterations=1000)
    bob = hash_password("pw-bob-1234", iterations=1000)
    users = parse_users(f"alice:{alice}, bob:{bob}")
    assert set(users) == {"alice", "bob"}
    assert authenticate(users, "alice", "pw-alice-123")
    assert not authenticate(users, "alice", "pw-bob-1234")
    assert not authenticate(users, "mallory", "anything")


def test_parse_users_rejects_bad_entries():
    with pytest.raises(ValueError):
        parse_users("alice")


# -- limits & commands -------------------------------------------------------


def test_rate_limiter_window():
    now = [0.0]
    limiter = RateLimiter(2, window_s=60, clock=lambda: now[0])
    assert limiter.check("u") == 0
    assert limiter.check("u") == 0
    assert limiter.check("u") == pytest.approx(60)
    assert limiter.check("other") == 0
    now[0] = 61
    assert limiter.check("u") == 0


def test_parse_command():
    assert parse_command("hello") is None
    assert parse_command("/unknown x") is None
    cmd = parse_command("  /BUILD  a todo app ")
    assert cmd.name == "build" and cmd.arg == "a todo app"
    assert parse_command("/docs").arg == ""


# -- streaming, retries, error mapping ---------------------------------------


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, script):
        self.script = list(script)  # each item: list of chunks, or an exception
        self.calls = 0

    async def stream(self, system, messages, usage):
        self.calls += 1
        step = self.script.pop(0)
        for item in step:
            if isinstance(item, BaseException):
                raise item
            yield item
        usage.output_tokens = len(step)


class Status(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


async def collect(provider, timeout=5):
    usage = Usage()
    out = [c async for c in stream_reply(provider, "sys", [ChatMessage("user", "hi")], usage, timeout)]
    return "".join(out), usage


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("nexora.providers.asyncio.sleep", no_sleep)


def test_stream_success():
    text, usage = asyncio.run(collect(FakeProvider([["Hel", "lo"]])))
    assert text == "Hello" and usage.output_tokens == 2


def test_stream_retries_transient_error_before_first_chunk():
    provider = FakeProvider([[Status(429)], [Status(503)], ["ok"]])
    text, _ = asyncio.run(collect(provider))
    assert text == "ok" and provider.calls == 3


def test_stream_does_not_retry_auth_errors():
    provider = FakeProvider([[Status(401)], ["never"]])
    with pytest.raises(ProviderError, match="API key"):
        asyncio.run(collect(provider))
    assert provider.calls == 1


def test_stream_does_not_retry_after_partial_output():
    provider = FakeProvider([["partial", Status(503)], ["again"]])

    async def run():
        chunks = []
        with pytest.raises(ProviderError):
            async for c in stream_reply(provider, "s", [], Usage(), 5):
                chunks.append(c)
        return chunks

    assert asyncio.run(run()) == ["partial"] and provider.calls == 1


def test_stream_timeout():
    class Slow:
        name, model = "slow", "s"

        async def stream(self, system, messages, usage):
            await asyncio.Event().wait()
            yield "never"

    async def run():
        async for _ in stream_reply(Slow(), "s", [], Usage(), timeout_s=0.05, max_attempts=1):
            pass

    with pytest.raises(ProviderError, match="too long"):
        asyncio.run(run())


@pytest.mark.parametrize(
    "exc,fragment,retryable",
    [
        (Status(404), "model name", False),
        (Status(400), "rejected the request", False),
        (Status(500), "server error", True),
        (ValueError("boom"), "Unexpected", False),
    ],
)
def test_classify(exc, fragment, retryable):
    err = classify(exc, "gemini")
    assert fragment in err.user_message and err.retryable is retryable


def test_help_text_hides_unavailable_features():
    from nexora.commands import help_text

    full = help_text()
    assert "/docs" in full and "/build" in full
    minimal = help_text(docs=False, build=False)
    assert "/docs" not in minimal and "/forget" not in minimal and "/build" not in minimal
    assert "/help" in minimal and "/usage" in minimal


# -- model fallback ----------------------------------------------------------


class NamedFake(FakeProvider):
    def __init__(self, model, script):
        super().__init__(script)
        self.model = model


class DailyQuota(Exception):
    status_code = 429

    def __str__(self):
        return "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def test_daily_quota_is_not_retried_but_falls_back():
    err = classify(DailyQuota(), "gemini")
    assert err.retryable is False and err.fallback is True
    assert classify(Status(429), "gemini").retryable is True
    assert classify(Status(401), "gemini").fallback is False


def test_stream_with_fallback_moves_to_next_model():
    from nexora.providers import stream_with_fallback

    first = NamedFake("gemini-a", [[DailyQuota()]])
    second = NamedFake("gemini-b", [["hi"]])

    async def run():
        usage = Usage()
        out = [c async for c in stream_with_fallback([first, second], "s", [], usage, 5)]
        return "".join(out), usage

    text, usage = asyncio.run(run())
    assert text == "hi" and usage.model == "gemini-b" and first.calls == 1


def test_stream_with_fallback_does_not_hide_auth_errors():
    from nexora.providers import stream_with_fallback

    first = NamedFake("gemini-a", [[Status(401)]])
    second = NamedFake("gemini-b", [["never"]])

    async def run():
        async for _ in stream_with_fallback([first, second], "s", [], Usage(), 5):
            pass

    with pytest.raises(ProviderError, match="API key"):
        asyncio.run(run())
    assert second.calls == 0


def test_candidate_models(env):
    from nexora.providers import candidate_models

    env.setenv("GEMINI_API_KEY", "k")
    s = config_mod.load_settings()
    assert candidate_models(s, "gemini", "") == ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.1-flash-lite"]
    assert candidate_models(s, "gemini", "my-model") == ["my-model"]
    env.setenv("GEMINI_FALLBACK_MODELS", "")
    assert candidate_models(config_mod.load_settings(), "gemini", None) == ["gemini-3.5-flash"]
