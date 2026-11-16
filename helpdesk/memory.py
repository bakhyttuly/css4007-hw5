"""User memory (Part 3): preferences that follow a USER across conversations.

Conversation memory (messages, steps tried) is NOT here: that is the LangGraph
checkpointer, keyed by thread_id. This module is the long-term store, keyed by
user_id. The user_id always comes from the application, never from the model.
"""
from __future__ import annotations

import re

_LANG_RE = re.compile(r"^[A-Za-z][A-Za-z \-]{1,39}$")


def namespace(user_id: str) -> tuple[str, str]:
    return ("preferences", user_id)


def save_language(store, user_id: str, language: str) -> dict:
    language = (language or "").strip()
    if not _LANG_RE.match(language):
        return {"error": "invalid_language", "note": "Give a language name such as 'Kazakh'."}
    store.put(namespace(user_id), "language", {"language": language.title()})
    return {"saved": True, "preference": "reply_language", "value": language.title()}


def load_user_memory(store, user_id: str) -> dict:
    item = store.get(namespace(user_id), "language")
    return {"language": item.value["language"]} if item else {}
