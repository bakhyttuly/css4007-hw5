"""Part 1 routing tests (no LLM, no API key needed)."""
import pytest

from helpdesk.data import load_helpdesk
from helpdesk.part1 import build_graph, run_with_trace

CASES_ORIGINAL = [
    ("I cannot connect to Wi-Fi.", "wifi", "wifi_help"),
    ("i cannot connect to wifi", "wifi", "wifi_help"),
    ("I forgot my PASSWORD.", "password", "password_help"),
    ("My laptop screen is broken.", "other", "other_help"),
    # both topics: original rule order sends it to Wi-Fi (wifi is checked first)
    ("My Wi-Fi password does not work.", "wifi", "wifi_help"),
]


@pytest.mark.parametrize("message,category,last_node", CASES_ORIGINAL)
def test_original_routing(message, category, last_node):
    res = run_with_trace(build_graph("original"), message)
    assert res["nodes_visited"] == ["read_request", last_node]
    assert res["final_state"]["category"] == category
    assert res["final_state"]["message"] == message  # original message untouched


def test_both_topics_go_to_clarify_after_change():
    res = run_with_trace(build_graph("with_clarification"), "My Wi-Fi password does not work.")
    assert res["nodes_visited"] == ["read_request", "clarify_problem"]
    assert res["final_state"]["category"] == "both"
    assert "Wi-Fi" in res["final_reply"] and "password" in res["final_reply"]


def test_clarified_graph_keeps_single_topic_routes():
    g = build_graph("with_clarification")
    assert run_with_trace(g, "I forgot my PASSWORD.")["nodes_visited"][-1] == "password_help"
    assert run_with_trace(g, "wifi is slow")["nodes_visited"][-1] == "wifi_help"
    assert run_with_trace(g, "broken screen")["nodes_visited"][-1] == "other_help"


def test_original_graph_has_no_clarify_node():
    assert "clarify_problem" not in build_graph("original").get_graph().nodes


def test_help_nodes_return_text_from_helpdesk_json():
    data = load_helpdesk()["instructions"]
    wifi = run_with_trace(build_graph(), "wifi")["final_reply"]
    pw = run_with_trace(build_graph(), "password")["final_reply"]
    assert all(step in wifi for step in data["wifi"]["steps"])
    assert all(step in pw for step in data["password"]["steps"])
