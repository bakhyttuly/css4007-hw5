"""The three helpdesk tools (Part 2).

Each tool is a plain Python function: it works when called directly, without
the agent. The agent only adds (a) a schema the model sees and (b) trusted
context (user, thread, status snapshot) that the MODEL cannot choose.
"""
from __future__ import annotations

import copy
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .data import DATA_DIR, OUTPUTS_DIR, load_helpdesk, load_status_profiles

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TICKETS_DIR = OUTPUTS_DIR / "tickets"

# crude guard: tickets must never contain a password / recovery code value
_SECRET_RE = re.compile(r"(password|passcode|recovery code)\s*(is|=|:)\s*\S+", re.I)


# the user made the ticket depend on the service being up ("if it is operating ...")
_CONDITIONAL_RE = re.compile(
    r"\bonly if\b|\bif\b[^.?!]{0,40}\b(operating|operational|working|up|running|available)\b",
    re.I)


_TICKET_WORD_RE = re.compile(r"\b(tickets?|drafts?)\b", re.I)


def ticket_was_requested(user_texts: list[str]) -> bool:
    """Has the USER asked for a ticket or draft anywhere in this conversation?"""
    return any(_TICKET_WORD_RE.search(t or "") for t in user_texts)


def ticket_is_conditional(user_text: str) -> bool:
    """Keyword heuristic on the user's own message (not on anything the model says)."""
    return bool(_CONDITIONAL_RE.search(user_text or ""))


# ----------------------------------------------------------------- plain functions
def find_instructions(topic: str) -> dict:
    """Look up Wi-Fi or password instructions in data/helpdesk.json."""
    entry = load_helpdesk()["instructions"].get(topic)
    if entry is None:
        return {"found": False, "topic": topic}
    return {
        "found": True,
        "topic": topic,
        "instruction_id": entry["instruction_id"],
        "title": entry["title"],
        "steps": entry["steps"],
        "source": "data/helpdesk.json",
    }


def check_service_status(service: str, status_profile: str) -> dict:
    """Read a service's state from the selected snapshot in data/service_status.json.

    'unknown' means the status source is unavailable: it is neither operating
    nor a confirmed outage.
    """
    profile = load_status_profiles()[status_profile]
    entry = profile.get(service)
    if entry is None:
        return {"service": service, "status": "unknown",
                "message": "No status information for this service.",
                "status_profile": status_profile}
    return {"service": service, "status": entry["status"], "message": entry["message"],
            "status_profile": status_profile, "source": "data/service_status.json"}


def prepare_ticket(category: str, summary: str, steps_tried: list[str],
                   service_status: str = "not_checked", *, user_id: str, thread_id: str,
                   status_profile: str | None = None,
                   tickets_dir: Path | str | None = None) -> dict:
    """Write a LOCAL ticket draft (data/ticket_template.json shape). Nothing is sent."""
    steps = [s.strip() for s in steps_tried if s and s.strip()]
    missing = []
    if not (summary or "").strip():
        missing.append("problem description")
    if not steps:
        missing.append("steps already tried")
    if missing:
        return {"error": "missing_details", "missing": missing,
                "note": "Ask the user for these details; do not invent them."}
    if _SECRET_RE.search(summary) or any(_SECRET_RE.search(s) for s in steps):
        return {"error": "secret_detected",
                "note": "Do not put passwords or recovery codes in a ticket."}

    template = json.loads((DATA_DIR / "ticket_template.json").read_text(encoding="utf-8"))
    ticket = copy.deepcopy(template["ticket"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ticket_id = f"TKT-{stamp}-{uuid.uuid4().hex[:4]}"
    ref = load_helpdesk()["instructions"].get(category, {}).get("instruction_id")
    ticket.update({
        "ticket_id": ticket_id,
        "status": "draft",
        "user_id": user_id,
        "thread_id": thread_id,
        "category": category,
        "summary": summary.strip(),
        "steps_tried": steps,
        "service_status": service_status,
        # snapshot is recorded only when the status was actually checked
        "status_profile": status_profile if service_status != "not_checked" else None,
        "reference_ids": [ref] if ref else [],
    })
    out_dir = Path(tickets_dir) if tickets_dir else DEFAULT_TICKETS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticket_id}.json"
    path.write_text(json.dumps({"ticket": ticket}, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    try:
        shown = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        shown = str(path)
    return {"ticket_id": ticket_id, "status": "draft", "path": shown,
            "delivery": "local draft only; nothing was sent"}


# ----------------------------------------------------------------- schemas shown to the model
class FindInstructionsArgs(BaseModel):
    topic: Literal["wifi", "password"] = Field(description="Which guide to look up.")


class CheckServiceStatusArgs(BaseModel):
    service: Literal["wifi", "account_recovery"] = Field(
        description="Service to check. Use account_recovery for password problems.")


class PrepareTicketArgs(BaseModel):
    category: Literal["wifi", "password"]
    summary: str = Field(description="Problem description, from the user's own words.")
    steps_tried: list[str] = Field(
        description="Steps the USER says they already tried. Never invent steps.")
    service_status: Literal["operational", "outage", "unknown", "not_checked"] = Field(
        default="not_checked", description="Result of check_service_status, if it was called.")
    only_if_operating: bool = Field(
        default=False,
        description="True when the user asked for the ticket ONLY IF the service is operating.")


class RememberLanguageArgs(BaseModel):
    language: str = Field(
        description="Reply language the user explicitly asked you to remember, e.g. 'Kazakh'.")


SCHEMAS: dict[str, type[BaseModel]] = {
    "find_instructions": FindInstructionsArgs,
    "check_service_status": CheckServiceStatusArgs,
    "prepare_ticket": PrepareTicketArgs,
    "remember_language_preference": RememberLanguageArgs,
}
DESCRIPTIONS = {
    "find_instructions": "Look up the Wi-Fi or password instructions in the local helpdesk guide.",
    "check_service_status": "Look up whether a service is operating, has a known outage, or its status is unknown.",
    "prepare_ticket": "Prepare a LOCAL ticket draft (nothing is sent). Needs a problem description and steps already tried.",
    "remember_language_preference": "Save the user's preferred reply language in long-term user memory. Only when the user explicitly asks you to remember it.",
}


def openai_tool_specs() -> list[dict]:
    return [
        {"type": "function",
         "function": {"name": n, "description": DESCRIPTIONS[n],
                      "parameters": SCHEMAS[n].model_json_schema()}}
        for n in SCHEMAS
    ]


# ----------------------------------------------------------------- dispatcher (used by the graph)
def execute_tool(name: str, args: dict, ctx: dict) -> dict:
    """Validate the model's request, then run the tool with TRUSTED context.

    ctx supplies user_id / thread_id / status_profile / tickets_dir. The model
    cannot override them: unknown extra arguments are ignored.
    """
    schema = SCHEMAS.get(name)
    if schema is None:
        return {"error": "unknown_tool", "tool": name}
    try:
        a = schema.model_validate(args)
    except ValidationError as e:
        return {"error": "invalid_arguments",
                "details": [{"field": ".".join(map(str, x["loc"])), "problem": x["msg"]}
                            for x in e.errors()]}
    if name == "remember_language_preference":
        from .memory import save_language
        return save_language(ctx["store"], ctx["user_id"], a.language)   # user_id is trusted context
    if name == "find_instructions":
        return find_instructions(a.topic)
    if name == "check_service_status":
        return check_service_status(a.service, ctx["status_profile"])
    if not ticket_was_requested(ctx.get("user_texts", [])):
        return {"error": "ticket_not_requested",
                "note": "The user has not asked for a ticket. Do not prepare one; help with "
                        "instructions or ask what they need."}
    # Valid arguments do not imply permission: the application re-reads the real status
    # from the selected snapshot instead of trusting what the model claims.
    service = load_helpdesk()["topic_services"][a.category]
    actual = check_service_status(service, ctx["status_profile"])["status"]
    conditional = a.only_if_operating or ticket_is_conditional(ctx.get("user_text", ""))
    if conditional and actual != "operational":
        return {"error": "condition_not_met", "service": service, "actual_status": actual,
                "note": "The user wanted a ticket only if the service is operating, but its "
                        f"status is '{actual}'. Do not prepare it. Tell the user; if the status "
                        "is unknown, say you cannot confirm it is operating and offer a draft "
                        "that states the status could not be checked."}
    recorded = "not_checked" if a.service_status == "not_checked" else actual
    return prepare_ticket(a.category, a.summary, a.steps_tried, recorded,
                          user_id=ctx["user_id"], thread_id=ctx["thread_id"],
                          status_profile=ctx.get("status_profile"),
                          tickets_dir=ctx.get("tickets_dir"))
