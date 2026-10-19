"""Parse slash commands typed into the chat box."""

from __future__ import annotations

from dataclasses import dataclass

COMMANDS = {
    "help": "Show this help",
    "docs": "List your uploaded documents",
    "forget": "Delete a document: /forget <number from /docs>",
    "reindex": "Re-embed all your documents (after changing the embedding model)",
    "build": "Run a MetaGPT software team: /build <project idea>",
    "usage": "Show token usage for this chat",
}


@dataclass
class Command:
    name: str
    arg: str


def parse_command(text: str) -> Command | None:
    text = text.strip()
    if not text.startswith("/"):
        return None
    name, _, arg = text[1:].partition(" ")
    name = name.lower()
    if name not in COMMANDS:
        return None
    return Command(name=name, arg=arg.strip())


DOC_COMMANDS = {"docs", "forget", "reindex"}


def help_text(docs: bool = True, build: bool = True) -> str:
    """List commands, leaving out features that aren't available to this user."""
    hidden = (set() if docs else DOC_COMMANDS) | (set() if build else {"build"})
    return "\n".join(f"- `/{name}` — {desc}" for name, desc in COMMANDS.items() if name not in hidden)
