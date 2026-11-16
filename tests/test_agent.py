"""Part 2 agent tests. They use the labelled FAKE model (no API key, not an LLM)."""
import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from helpdesk.agent import build_agent, make_config, run_turn
from helpdesk.fake_model import FakeRuleModel

CONDITIONAL = ("I enabled Wi-Fi and forgot the saved network, but Campus-Demo still will not "
               "connect. Check whether Wi-Fi is down. If it is operating, prepare a ticket "
               "about this problem.")


def make(tmp_path, model=None, **kw):
    return build_agent(model or FakeRuleModel(), checkpointer=InMemorySaver(),
                       store=InMemoryStore(), tickets_dir=tmp_path, **kw)


def ask(agent, text, profile="normal", thread="t1", user="demo-user-a"):
    return run_turn(agent, text, make_config(user, thread, profile))


def test_operational_leads_to_ticket(tmp_path):
    r = ask(make(tmp_path), CONDITIONAL, "normal")
    assert r["tools_used"] == ["check_service_status", "prepare_ticket"]
    assert len(list(tmp_path.glob("TKT-*.json"))) == 1 and r["tickets"]
    assert "Nothing has been sent" in r["final_reply"]


def test_outage_means_no_ticket(tmp_path):
    r = ask(make(tmp_path), CONDITIONAL, "wifi_outage")
    assert r["tools_used"] == ["check_service_status"]
    assert list(tmp_path.glob("*.json")) == [] and "outage" in r["final_reply"]
    assert "repair time" in r["final_reply"]


def test_unknown_status_means_no_ticket_and_no_claim(tmp_path):
    r = ask(make(tmp_path), CONDITIONAL, "status_unavailable")
    assert r["tools_used"] == ["check_service_status"] and list(tmp_path.glob("*.json")) == []
    assert "cannot confirm" in r["final_reply"]


def test_password_topic_checks_account_recovery(tmp_path):
    r = ask(make(tmp_path), "My password recovery page is unavailable. Check the service status. "
            "Is there a known problem?", "recovery_outage")
    status_call = [t for t in r["trace"] if t["node"] == "tools"][0]["update"]["messages"][0]
    assert status_call["result"]["service"] == "account_recovery"
    assert status_call["result"]["status"] == "outage"


def test_missing_ticket_details_lead_to_a_question(tmp_path):
    agent = make(tmp_path)
    r1 = ask(agent, "Prepare a support ticket.", thread="s09")
    assert r1["tools_used"] == [] and "What problem" in r1["final_reply"]
    assert list(tmp_path.glob("*.json")) == []
    r2 = ask(agent, "The issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.",
             thread="s09")
    assert r2["tools_used"] == ["prepare_ticket"] and len(list(tmp_path.glob("*.json"))) == 1


def test_model_cannot_choose_the_status_snapshot(tmp_path):
    class Sneaky(FakeRuleModel):
        def invoke(self, messages):
            if any(type(m).__name__ == "ToolMessage" for m in messages[-1:]):
                return AIMessage(content="done")
            return self._call(messages, "check_service_status",
                              {"service": "wifi", "status_profile": "normal"})
    r = ask(make(tmp_path, Sneaky()), "hello", "wifi_outage")
    res = [t for t in r["trace"] if t["node"] == "tools"][0]["update"]["messages"][0]["result"]
    assert res["status"] == "outage"      # the snapshot chosen by the application won


def test_loop_limit_stops_a_model_that_never_stops(tmp_path):
    class Forever(FakeRuleModel):
        def invoke(self, messages):
            return self._call(messages, "find_instructions", {"topic": "wifi"})
    agent = make(tmp_path, Forever(), max_tool_calls=3, max_model_calls=10)
    r = ask(agent, "hi")
    assert r["loop_limit_reached"] and r["nodes_visited"][-1] == "limit"
    assert r["tools_used"].count("find_instructions") == 4   # 3 real + 1 refused with an error
    assert "limit" in r["final_reply"]


def test_budget_resets_for_each_new_user_turn(tmp_path):
    class Forever(FakeRuleModel):
        def invoke(self, messages):
            return self._call(messages, "find_instructions", {"topic": "wifi"})
    agent = make(tmp_path, Forever(), max_tool_calls=2)
    assert ask(agent, "one")["loop_limit_reached"]
    second = ask(agent, "two")
    assert second["tools_used"].count("find_instructions") >= 3   # budget was reset, not kept at 2


def test_conversation_history_accumulates_in_one_thread(tmp_path):
    agent = make(tmp_path)
    ask(agent, "I enabled Wi-Fi and reconnected but it fails.", thread="s10")
    r = ask(agent, "Which steps did I already try?", thread="s10")
    assert "Enabled Wi-Fi" in r["final_reply"]
    other = ask(agent, "Which steps did I already try?", thread="other")
    assert "no earlier messages" in other["final_reply"]


class DisobedientModel(FakeRuleModel):
    """Ignores the 'only if operating' rule and always tries to prepare a ticket."""
    def invoke(self, messages):
        if type(messages[-1]).__name__ == "ToolMessage":
            return AIMessage(content="done")
        return self._call(messages, "prepare_ticket", {
            "category": "wifi", "summary": "Wi-Fi fails", "steps_tried": ["Enabled Wi-Fi"],
            "service_status": "operational", "only_if_operating": True})


@pytest.mark.parametrize("profile,expected", [
    ("wifi_outage", "outage"), ("status_unavailable", "unknown")])
def test_code_blocks_conditional_ticket_even_if_model_ignores_the_rule(tmp_path, profile, expected):
    r = ask(make(tmp_path, DisobedientModel()), "ticket only if wifi is up", profile)
    res = [t for t in r["trace"] if t["node"] == "tools"][0]["update"]["messages"][0]["result"]
    assert res["error"] == "condition_not_met" and res["actual_status"] == expected
    assert list(tmp_path.glob("*.json")) == []


def test_ticket_records_real_status_not_the_models_claim(tmp_path):
    class Liar(FakeRuleModel):
        def invoke(self, messages):
            if type(messages[-1]).__name__ == "ToolMessage":
                return AIMessage(content="done")
            return self._call(messages, "prepare_ticket", {
                "category": "wifi", "summary": "Wi-Fi fails", "steps_tried": ["Enabled Wi-Fi"],
                "service_status": "operational"})
    ask(make(tmp_path, Liar()), "make a ticket", "wifi_outage")
    saved = json.loads(next(tmp_path.glob("TKT-*.json")).read_text(encoding="utf-8"))["ticket"]
    assert saved["service_status"] == "outage"


class OmitsFlagModel(FakeRuleModel):
    """Like the real gpt-4o-mini in S08: prepares a ticket and never sets only_if_operating."""
    def invoke(self, messages):
        if type(messages[-1]).__name__ == "ToolMessage":
            return AIMessage(content="done")
        return self._call(messages, "prepare_ticket", {
            "category": "wifi", "summary": "Wi-Fi fails", "steps_tried": ["Enabled Wi-Fi"]})


def test_condition_is_read_from_the_users_message_not_from_a_model_flag(tmp_path):
    text = "Check whether Wi-Fi is down. If it is operating, prepare a ticket."
    r = ask(make(tmp_path, OmitsFlagModel()), text, "status_unavailable")
    res = [t for t in r["trace"] if t["node"] == "tools"][0]["update"]["messages"][0]["result"]
    assert res["error"] == "condition_not_met" and list(tmp_path.glob("*.json")) == []


def test_unconditional_ticket_request_is_still_allowed_during_outage(tmp_path):
    r = ask(make(tmp_path, OmitsFlagModel()), "Please prepare a ticket for my Wi-Fi.", "wifi_outage")
    assert r["tickets"] and len(list(tmp_path.glob("*.json"))) == 1


def test_ticket_is_conditional_heuristic():
    from helpdesk.tools import ticket_is_conditional
    assert ticket_is_conditional("If it is operating, prepare a ticket.")
    assert ticket_is_conditional("Only if the service is up.")
    assert not ticket_is_conditional("Prepare a support ticket.")
    assert not ticket_is_conditional("I forgot the saved network, but it still will not connect.")


def test_code_refuses_a_ticket_the_user_never_asked_for(tmp_path):
    r = ask(make(tmp_path, OmitsFlagModel()),
            "I cannot connect to Wi-Fi. I enabled it and forgot the saved network.", "normal")
    res = [t for t in r["trace"] if t["node"] == "tools"][0]["update"]["messages"][0]["result"]
    assert res["error"] == "ticket_not_requested" and list(tmp_path.glob("*.json")) == []


def test_ticket_request_earlier_in_the_thread_still_counts(tmp_path):
    agent = make(tmp_path, OmitsFlagModel())
    ask(agent, "Could you prepare a support ticket?", thread="x")       # request, no details yet
    r = ask(agent, "The issue is Wi-Fi, I enabled it.", thread="x")
    assert r["tickets"]
