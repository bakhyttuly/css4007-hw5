"""Part 2: model + tools graph (ReAct loop with an enforced limit).

    START -> start_turn -> load_user_memory -> model --(tool requested?)--> tools -> model ...
                             |  no tool call  -> END
                             |  budget exceeded -> limit -> END

* `model`  requests tools (it never executes anything).
* `tools`  validates each request and executes it with trusted context.
* The loop limit is enforced in `after_model` (a routing function in OUR code),
  so a model that keeps asking for tools cannot run forever.
"""
from __future__ import annotations

import json
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .memory import load_user_memory
from .tools import execute_tool, openai_tool_specs

MAX_MODEL_CALLS = 8   # model invocations per user turn
MAX_TOOL_CALLS = 6    # tool executions per user turn

SYSTEM_PROMPT = """You are a university IT helpdesk assistant (practice exercise, fictional data).
Scope: Wi-Fi troubleshooting, password recovery guidance, service status lookup, and preparing LOCAL ticket drafts.

Rules:
- Get instructions only with find_instructions. Never invent troubleshooting steps.
- Never claim a service is up or down without check_service_status. Password problems use the account_recovery service.
- If the user makes a ticket conditional on the service status, follow the tool result:
  operational -> prepare the ticket; outage -> do NOT prepare it, report the outage and say no repair time is given;
  unknown -> do NOT claim the service is operating, say the status could not be determined and offer a draft that says so.
- Do NOT prepare a ticket unless the user asks for one. If they only describe a problem, give instructions (find_instructions) and mention that you can prepare a ticket if it is still not solved.
- Call prepare_ticket only with a problem description and steps the USER says they already tried. If either is missing, ask the user; do not invent details.
- Once the user has asked for a ticket (in this or an earlier message) AND you have both a problem description and the steps they tried, call prepare_ticket right away. Do not ask whether the ticket should be conditional or whether to check the status unless the user said so; with no stated condition, leave service_status as not_checked.
- steps_tried must contain ONLY what the user said they did, in their own words. Do not copy steps from the instructions unless the user said they did them. Never add steps such as "airplane mode is off" on your own.
- When the user wants a ticket only if the service is operating, set only_if_operating=true. If prepare_ticket returns condition_not_met, do not retry; follow its note.
- Tickets are local drafts. Say that nothing has been sent, and give the ticket filename.
- Never ask for or store passwords or recovery codes. Never invent repair times or real outages.
- Save a reply-language preference with remember_language_preference ONLY when the user explicitly asks you to remember it.
- Answer questions about earlier problems or saved preferences only from this conversation and the "User memory" line below. If neither contains it, say that no preference or previous problem has been recorded. Never guess.
- Be concise."""


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # reducer: new messages are appended
    model_calls: int
    tool_calls: int
    stop_reason: str | None
    user_memory: dict     # read from the user-memory store at the start of every turn


def build_agent(model, *, checkpointer=None, store=None, tickets_dir=None,
                max_model_calls: int = MAX_MODEL_CALLS, max_tool_calls: int = MAX_TOOL_CALLS):
    bound = model.bind_tools(openai_tool_specs())

    def start_turn(state: AgentState) -> dict:
        # budgets are per user turn: reset them every time a new message arrives
        return {"model_calls": 0, "tool_calls": 0, "stop_reason": None}

    def load_memory(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
        # long-term memory: keyed by the user_id our application put in the config
        user_id = config["configurable"]["user_id"]
        return {"user_memory": load_user_memory(store, user_id)}

    def call_model(state: AgentState) -> dict:
        lang = (state.get("user_memory") or {}).get("language")
        mem = (f"User memory (saved for this user in earlier conversations): reply language = {lang}. "
               f"Reply in {lang}." if lang
               else "User memory: nothing is saved for this user.")
        reply = bound.invoke([SystemMessage(SYSTEM_PROMPT + "\n\n" + mem)] + state["messages"])
        return {"messages": [reply], "model_calls": state.get("model_calls", 0) + 1}

    def after_model(state: AgentState) -> str:
        calls = getattr(state["messages"][-1], "tool_calls", None) or []
        if not calls:
            return END
        if (state.get("tool_calls", 0) + len(calls) > max_tool_calls
                or state.get("model_calls", 0) >= max_model_calls):
            return "limit"
        return "tools"

    def run_tools(state: AgentState, config: RunnableConfig, *, store: BaseStore) -> dict:
        ctx = dict(config.get("configurable", {}))
        ctx["tickets_dir"] = tickets_dir
        ctx["store"] = store
        # the user's latest words, used by tools to enforce conditions the user stated
        ctx["user_texts"] = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
        ctx["user_text"] = ctx["user_texts"][-1] if ctx["user_texts"] else ""
        calls = state["messages"][-1].tool_calls
        out = []
        for call in calls:
            result = execute_tool(call["name"], call["args"], ctx)
            out.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                   tool_call_id=call["id"], name=call["name"]))
        return {"messages": out, "tool_calls": state.get("tool_calls", 0) + len(calls)}

    def limit(state: AgentState) -> dict:
        # every pending tool request needs an answer, then we stop with a clear message
        pending = state["messages"][-1].tool_calls
        out = [ToolMessage(content=json.dumps({"error": "loop_limit_reached"}),
                           tool_call_id=c["id"], name=c["name"]) for c in pending]
        out.append(AIMessage(content="I stopped because I reached the tool-call limit for "
                                     "this request. Please try again with a simpler request."))
        return {"messages": out, "stop_reason": "loop_limit"}

    b = StateGraph(AgentState)
    b.add_node("start_turn", start_turn)
    b.add_node("load_user_memory", load_memory)
    b.add_node("model", call_model)
    b.add_node("tools", run_tools)
    b.add_node("limit", limit)
    b.add_edge(START, "start_turn")
    b.add_edge("start_turn", "load_user_memory")
    b.add_edge("load_user_memory", "model")
    b.add_conditional_edges("model", after_model, {"tools": "tools", "limit": "limit", END: END})
    b.add_edge("tools", "model")
    b.add_edge("limit", END)
    return b.compile(checkpointer=checkpointer, store=store)


# ----------------------------------------------------------------- running + tracing
def make_config(user_id: str, thread_id: str, status_profile: str) -> dict:
    """thread_id selects the conversation (checkpointer); user_id and status_profile
    are chosen by OUR application, never by the model."""
    return {"configurable": {"thread_id": thread_id, "user_id": user_id,
                             "status_profile": status_profile}}


def _jsonable(update: dict) -> dict:
    out = {}
    for k, v in update.items():
        if k == "messages":
            msgs = []
            for m in v:
                if isinstance(m, ToolMessage):
                    msgs.append({"type": "tool_result", "tool": m.name,
                                 "result": json.loads(m.content)})
                else:
                    msgs.append({"type": "ai", "content": m.content,
                                 "tool_requests": [{"name": c["name"], "args": c["args"]}
                                                   for c in (getattr(m, "tool_calls", None) or [])]})
            out[k] = msgs
        else:
            out[k] = v
    return out


def run_turn(agent, text: str, config: dict) -> dict:
    """Send one user message; return trace, final reply and tickets written."""
    trace, final_reply, tools_used, tickets, limit_hit = [], None, [], [], False
    for chunk in agent.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
        for node, update in chunk.items():
            trace.append({"node": node, "update": _jsonable(update)})
            for m in update.get("messages", []):
                if isinstance(m, ToolMessage):
                    tools_used.append(m.name)
                    res = json.loads(m.content)
                    if m.name == "prepare_ticket" and "path" in res:
                        tickets.append(res["path"])
                elif isinstance(m, AIMessage) and not m.tool_calls:
                    final_reply = m.content
            if update.get("stop_reason") == "loop_limit":
                limit_hit = True
    mem = next((t["update"]["user_memory"] for t in trace if t["node"] == "load_user_memory"), {})
    return {"message": text, "user_memory_read": mem, "nodes_visited": [t["node"] for t in trace], "trace": trace,
            "tools_used": tools_used, "final_reply": final_reply,
            "tickets": tickets, "loop_limit_reached": limit_hit}
