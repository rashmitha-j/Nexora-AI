"""Password sign-in against PBKDF2 hashes kept in the NEXORA_USERS env var.

NEXORA_USERS format: "alice:<hash>,bob:<hash>". Create a hash with:
    python -m nexora.auth
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import secrets
import time

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 600_000


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"{ALGORITHM}${iterations}${b64(salt)}${b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = encoded.split("$")
        if algorithm != ALGORITHM:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def clean_env_value(raw: str, name: str) -> str:
    """Undo common copy-paste mistakes in dashboard env vars: surrounding whitespace or
    quotes, and a leading "NAME=" copied together with the value."""
    value = (raw or "").strip()
    if value.upper().startswith(f"{name.upper()}="):
        value = value[len(name) + 1 :].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def parse_users_lenient(spec: str) -> tuple[dict[str, str], list[str]]:
    """Like parse_users, but skips bad entries and reports them instead of failing, so a
    typo in one account can't take the whole site down. Errors never include secrets."""
    users: dict[str, str] = {}
    errors: list[str] = []
    for n, entry in enumerate(clean_env_value(spec, "NEXORA_USERS").split(","), start=1):
        entry = entry.strip().strip("\"'")
        if not entry:
            continue
        name, sep, encoded = entry.partition(":")
        name, encoded = name.strip(), encoded.strip()
        if not sep or not name or not encoded.startswith(f"{ALGORITHM}$") or encoded.count("$") != 3:
            errors.append(f"entry {n} is not 'username:{ALGORITHM}$...' (create it with: python -m nexora.auth)")
            continue
        users[name] = encoded
    return users, errors


class LoginThrottle:
    """Refuse further attempts for a username after too many failures in a time window."""

    def __init__(self, max_failures: int = 5, window_s: float = 900, clock=time.monotonic):
        self.max_failures = max_failures
        self.window_s = window_s
        self._clock = clock
        self._failures: dict[str, list[float]] = {}

    def _recent(self, username: str) -> list[float]:
        now = self._clock()
        recent = [t for t in self._failures.get(username, []) if now - t < self.window_s]
        self._failures[username] = recent
        return recent

    def blocked(self, username: str) -> bool:
        return len(self._recent(username)) >= self.max_failures

    def record_failure(self, username: str) -> None:
        self._recent(username).append(self._clock())
        if len(self._failures) > 10_000:  # bound memory against random-username floods
            for name in list(self._failures):
                if not self._recent(name):
                    del self._failures[name]

    def reset(self, username: str) -> None:
        self._failures.pop(username, None)


def parse_users(spec: str) -> dict[str, str]:
    users: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, sep, encoded = entry.partition(":")
        if not sep or not name.strip() or not encoded.strip():
            raise ValueError("NEXORA_USERS entries must look like 'username:pbkdf2_sha256$...'")
        users[name.strip()] = encoded.strip()
    return users


def authenticate(users: dict[str, str], username: str, password: str) -> bool:
    encoded = users.get(username)
    if encoded is None:
        # Spend similar time for unknown users so timing doesn't reveal valid names.
        verify_password(password, hash_password("dummy", iterations=ITERATIONS))
        return False
    return verify_password(password, encoded)


if __name__ == "__main__":
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    if len(password) < 10:
        raise SystemExit("Use a password of at least 10 characters.")
    print(f"{username}:{hash_password(password)}")
