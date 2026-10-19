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
