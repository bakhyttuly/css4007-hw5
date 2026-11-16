"""FAKE model for tests and CI. It is NOT an LLM.

A small deterministic rule-based stand-in that requests tools and answers from
tool results, so the graph, tools and memory can be tested without an API key.
Real-model demonstrations must NOT use it (logs from it are labelled 'fake').
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

FAKE_INFO = {"provider": "fake", "model": "FakeRuleModel (deterministic test double, NOT an LLM)"}


def derive_steps(text: str) -> list[str]:
    t = text.lower()
    steps = []
    if "enabled" in t:
        steps.append("Enabled Wi-Fi")
    if "forgot the saved network" in t:
        steps.append("Forgot the saved network")
    if "reconnect" in t:
        steps.append("Tried reconnecting")
    return steps


class FakeRuleModel:
    def bind_tools(self, tools):
        return self

    @staticmethod
    def _call(msgs, name, args):
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": f"call_{len(msgs)}_{name}", "type": "tool_call"}])

    def invoke(self, messages):
        msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        last_h = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
        results = {m.name: json.loads(m.content) for m in msgs[last_h + 1:]
                   if isinstance(m, ToolMessage)}
        humans = [m.content.lower() for m in msgs if isinstance(m, HumanMessage)]
        now, all_text = humans[-1], " ".join(humans)
        earlier = " ".join(humans[:-1])
        topic = "password" if "password" in now else "wifi"
        service = "account_recovery" if topic == "password" else "wifi"

        sys_text = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
        saved = re.search(r"reply language = (\w+)", sys_text)
        saved_lang = saved.group(1) if saved else None

        if "remember" in now and "prefer" in now:
            if "remember_language_preference" not in results:
                lang = re.search(r"in (\w+)", now.split("prefer", 1)[1])
                return self._call(msgs, "remember_language_preference",
                                  {"language": lang.group(1) if lang else ""})
            r = results["remember_language_preference"]
            return AIMessage(content=f"Saved: I will reply in {r.get('value')}." if "saved" in r
                             else "I could not save that preference.")

        if "which language" in now or "preferred language" in now:
            if saved_lang:
                return AIMessage(content=f"You asked me to reply in {saved_lang}.")
            return AIMessage(content="No preference or previous problem has been recorded for you.")

        if "already try" in now or "which problem" in now or "which steps" in now:
            steps = derive_steps(earlier)
            if not steps:
                return AIMessage(content="I have no earlier messages in this conversation.")
            return AIMessage(content="You reported a Wi-Fi problem and already tried: "
                                     + "; ".join(steps) + ".")

        if "service status" in now or "is down" in now or "check whether" in now:
            if "check_service_status" not in results:
                return self._call(msgs, "check_service_status", {"service": service})
            st = results["check_service_status"]
            status = st["status"]
            conditional = "ticket" in now and "if it is operating" in now
            if conditional and status == "operational":
                if "prepare_ticket" not in results:
                    steps = derive_steps(all_text)
                    summary = ("Wi-Fi: cannot connect to Campus-Demo after the steps already tried"
                               if "campus-demo" in now else "Wi-Fi connection still fails")
                    return self._call(msgs, "prepare_ticket", {
                        "category": "wifi", "summary": summary, "steps_tried": steps,
                        "service_status": "operational",
                        "only_if_operating": True})
                r = results["prepare_ticket"]
                if "error" in r:
                    return AIMessage(content="What problem are you having, and what have you already tried?")
                return AIMessage(content=f"The demo status source reports {service} is operating. "
                                         f"I prepared a local draft ({r['path']}) describing the failed "
                                         "connection and the steps you already tried. Nothing has been sent.")
            if status == "operational":
                return AIMessage(content=st["message"])
            if status == "outage":
                extra = (" Your request said to prepare a ticket if the service was operating, "
                         "so I have not prepared that ticket." if conditional else "")
                return AIMessage(content="The demo status source reports an outage for "
                                         f"{service}.{extra} The data does not give a repair time.")
            return AIMessage(content="I could not determine the status. I cannot confirm that it is "
                                     "operating. Would you like a ticket draft explaining that the "
                                     "status could not be checked?")

        if "steps" in now or "instructions" in now:
            if "find_instructions" not in results:
                return self._call(msgs, "find_instructions", {"topic": topic})
            r = results["find_instructions"]
            return AIMessage(content=f"[{r['instruction_id']}] " + " ".join(r["steps"]))

        if "ticket" in all_text:
            steps = derive_steps(all_text)
            if "wi-fi" not in all_text and "wifi" not in all_text and "password" not in all_text or not steps:
                return AIMessage(content="What problem are you having, and what have you already tried?")
            if "prepare_ticket" not in results:
                return self._call(msgs, "prepare_ticket", {
                    "category": "wifi", "summary": "Wi-Fi connection still fails",
                    "steps_tried": steps, "service_status": "not_checked"})
            r = results["prepare_ticket"]
            return AIMessage(content=f"I prepared a local draft ({r['path']}). The service status "
                                     "was not checked. Nothing has been sent.")

        return AIMessage(content="I can help with Wi-Fi, password recovery, service status and "
                                 "local ticket drafts. What do you need?")
