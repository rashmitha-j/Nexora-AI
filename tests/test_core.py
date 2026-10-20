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


def test_config_auth_no_longer_requires_secret_at_validation(env):
    # The app supplies a temporary secret instead of crashing (see nexora_app.py).
    env.setenv("GEMINI_API_KEY", "k")
    env.setenv("NEXORA_USERS", "a:x")
    assert config_mod.load_settings().auth_enabled


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
def fresh_quota_memory():
    from nexora import providers as prov

    prov._exhausted_until.clear()
    yield
    prov._exhausted_until.clear()


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("nexora.providers.asyncio.sleep", no_sleep)


def test_stream_success():
    text, usage = asyncio.run(collect(FakeProvider([["Hel", "lo"]])))
    assert text == "Hello" and usage.output_tokens == 2


def test_stream_retries_transient_error_before_first_chunk():
    provider = FakeProvider([[Status(429)], [Status(502)], ["ok"]])
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
    from nexora.providers import allowed_models, candidate_models

    env.setenv("GEMINI_API_KEY", "k")
    s = config_mod.load_settings()
    chain = [
        "gemini-3.5-flash"
    ] + "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite".split(",")
    assert allowed_models(s, "gemini") == chain
    assert candidate_models(s, "gemini", "") == chain
    # An allowed choice goes first and keeps the others as fallbacks.
    assert candidate_models(s, "gemini", "gemini-3.7-flash")[0] == "gemini-3.7-flash"
    assert sorted(candidate_models(s, "gemini", "gemini-3.7-flash")) == sorted(chain)
    # Anything else a client sends (e.g. a costlier model) is ignored.
    assert candidate_models(s, "gemini", "gemini-3.1-pro-preview") == chain
    env.setenv("GEMINI_FALLBACK_MODELS", "")
    assert candidate_models(config_mod.load_settings(), "gemini", None) == ["gemini-3.5-flash"]


class DailyQuotaWithDelay(DailyQuota):
    def __str__(self):
        return super().__str__() + ' "retryDelay": "64138s"'


def test_global_rate_limit_setting(env):
    env.setenv("GEMINI_API_KEY", "k")
    assert config_mod.load_settings().global_rate_limit_per_minute == 30
    env.setenv("GLOBAL_RATE_LIMIT_PER_MINUTE", "5")
    assert config_mod.load_settings().global_rate_limit_per_minute == 5


def test_exhausted_model_is_skipped_until_reset():
    from nexora import providers as prov

    prov._exhausted_until.clear()
    first = NamedFake("gemini-a", [[DailyQuotaWithDelay()], ["should not be called again"]])
    second = NamedFake("gemini-b", [["one"], ["two"]])

    async def ask():
        usage = Usage()
        out = [c async for c in prov.stream_with_fallback([first, second], "s", [], usage, 5)]
        return "".join(out), usage.model

    assert asyncio.run(ask()) == ("one", "gemini-b")
    assert asyncio.run(ask()) == ("two", "gemini-b")
    assert first.calls == 1  # skipped on the second message
    assert classify(DailyQuotaWithDelay(), "gemini").retry_after_s == 64138.0
    # When every model is exhausted, the last one is still tried so the user gets an answer.
    assert [p.model for p in prov.available([first])] == ["gemini-a"]
    prov._exhausted_until.clear()


# -- sign-in hardening -------------------------------------------------------


def test_clean_env_value_undoes_paste_mistakes():
    from nexora.auth import clean_env_value

    assert clean_env_value('  "abc"  ', "CHAINLIT_AUTH_SECRET") == "abc"
    assert clean_env_value('CHAINLIT_AUTH_SECRET="abc"', "CHAINLIT_AUTH_SECRET") == "abc"
    assert clean_env_value("abc", "CHAINLIT_AUTH_SECRET") == "abc"
    assert clean_env_value("", "X") == ""


def test_parse_users_lenient_skips_bad_entries_without_leaking():
    from nexora.auth import parse_users_lenient

    good = hash_password("pw-alice-123", iterations=1000)
    users, errors = parse_users_lenient(f'"alice:{good}", bob:not-a-hash, , carol')
    assert users == {"alice": good}
    assert len(errors) == 2 and all("not-a-hash" not in e for e in errors)
    assert authenticate(users, "alice", "pw-alice-123")


def test_login_throttle_blocks_after_repeated_failures():
    from nexora.auth import LoginThrottle

    now = [0.0]
    t = LoginThrottle(max_failures=3, window_s=60, clock=lambda: now[0])
    for _ in range(3):
        assert not t.blocked("alice")
        t.record_failure("alice")
    assert t.blocked("alice") and not t.blocked("bob")
    now[0] = 61
    assert not t.blocked("alice")
    t.record_failure("alice")
    t.reset("alice")
    assert not t.blocked("alice")


def test_overload_falls_back_without_retrying_same_model():
    err = classify(Status(503), "gemini")
    assert err.retryable is False and err.fallback is True
    assert classify(Status(500), "gemini").retryable is True
