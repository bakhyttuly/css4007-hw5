"""Part 1: a helpdesk graph that decides WITHOUT an LLM.

Graph (original):
    START -> read_request -> [conditional edge] -> wifi_help | password_help | other_help -> END

Graph (with_clarification) adds one more branch:
    ... -> clarify_problem -> END      (message mentions BOTH Wi-Fi and password)

Division of labour:
  * LangGraph: runs nodes in order, merges each node's returned update into the
    shared state, calls the routing function and follows the edge it names.
  * Our Python functions: decide the category (detect_topics / categorize_*),
    pick the branch name (route_after_read) and write the reply text.
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .data import load_helpdesk

Variant = Literal["original", "with_clarification"]

CLARIFY_TEXT = "Is the problem connecting to Wi-Fi, or recovering your account password?"
OTHER_TEXT = (
    "I can only help with Wi-Fi or password problems. "
    "Is your problem about Wi-Fi or a password?"
)


class HelpdeskState(TypedDict, total=False):
    message: str   # the user's original message (never modified)
    category: str  # set by read_request: wifi | password | both | other
    reply: str     # set by one of the help nodes


# ---------------------------------------------------------------- plain Python rules
def detect_topics(message: str, aliases: dict[str, list[str]]) -> set[str]:
    """Return the topics whose alias appears in the message (case-insensitive)."""
    text = message.lower()
    return {topic for topic, words in aliases.items() if any(w in text for w in words)}


def categorize_original(topics: set[str]) -> str:
    """Original rule: wifi wins, then password, else other."""
    if "wifi" in topics:
        return "wifi"
    if "password" in topics:
        return "password"
    return "other"


def categorize_clarified(topics: set[str]) -> str:
    """Changed rule: both topics -> 'both' (goes to clarify_problem)."""
    if {"wifi", "password"} <= topics:
        return "both"
    return categorize_original(topics)


def format_instructions(entry: dict) -> str:
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(entry["steps"], 1))
    return f"{entry['title']} [{entry['instruction_id']}]\n{steps}"


# ---------------------------------------------------------------- graph builder
def build_graph(variant: Variant = "original"):
    data = load_helpdesk()
    aliases = data["topic_aliases"]
    instructions = data["instructions"]
    categorize = categorize_clarified if variant == "with_clarification" else categorize_original

    # --- nodes: read the state, return ONLY the fields they change
    def read_request(state: HelpdeskState) -> dict:
        topics = detect_topics(state["message"], aliases)
        return {"category": categorize(topics)}

    def wifi_help(state: HelpdeskState) -> dict:
        return {"reply": format_instructions(instructions["wifi"])}

    def password_help(state: HelpdeskState) -> dict:
        return {"reply": format_instructions(instructions["password"])}

    def other_help(state: HelpdeskState) -> dict:
        return {"reply": OTHER_TEXT}

    def clarify_problem(state: HelpdeskState) -> dict:
        return {"reply": CLARIFY_TEXT}

    # --- conditional edge: returns the NAME of the next node, changes no state
    def route_after_read(state: HelpdeskState) -> str:
        return {
            "wifi": "wifi_help",
            "password": "password_help",
            "both": "clarify_problem",
        }.get(state["category"], "other_help")

    branches = ["wifi_help", "password_help", "other_help"]
    if variant == "with_clarification":
        branches.append("clarify_problem")

    builder = StateGraph(HelpdeskState)
    builder.add_node("read_request", read_request)
    builder.add_node("wifi_help", wifi_help)
    builder.add_node("password_help", password_help)
    builder.add_node("other_help", other_help)
    if variant == "with_clarification":
        builder.add_node("clarify_problem", clarify_problem)

    builder.add_edge(START, "read_request")
    builder.add_conditional_edges("read_request", route_after_read, {b: b for b in branches})
    for b in branches:
        builder.add_edge(b, END)
    return builder.compile()


# ---------------------------------------------------------------- tracing
def run_with_trace(graph, message: str) -> dict:
    """Run the graph and record every node and the full state after it."""
    state: dict = {"message": message}
    trace = []
    for chunk in graph.stream({"message": message}, stream_mode="updates"):
        for node, update in chunk.items():
            state = {**state, **update}
            trace.append({"node": node, "update": update, "state_after": dict(state)})
    return {
        "message": message,
        "nodes_visited": [t["node"] for t in trace],
        "trace": trace,
        "final_state": state,
        "final_reply": state.get("reply"),
    }
