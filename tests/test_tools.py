"""The three tools work when called directly (no agent, no model)."""
import json

import pytest

from helpdesk.tools import (check_service_status, execute_tool, find_instructions,
                            prepare_ticket)


def test_find_instructions_returns_guide_from_json():
    r = find_instructions("wifi")
    assert r["found"] and r["instruction_id"] == "GUIDE-WIFI" and len(r["steps"]) == 4
    assert find_instructions("password")["instruction_id"] == "GUIDE-PASSWORD"


def test_find_instructions_unknown_topic():
    assert find_instructions("printer") == {"found": False, "topic": "printer"}


@pytest.mark.parametrize("profile,service,status", [
    ("normal", "wifi", "operational"),
    ("wifi_outage", "wifi", "outage"),
    ("wifi_outage", "account_recovery", "operational"),
    ("recovery_outage", "account_recovery", "outage"),
    ("status_unavailable", "wifi", "unknown"),
])
def test_check_service_status_uses_selected_snapshot(profile, service, status):
    r = check_service_status(service, profile)
    assert r["status"] == status and r["status_profile"] == profile


def test_prepare_ticket_writes_local_draft(tmp_path):
    r = prepare_ticket("wifi", "Cannot connect to Campus-Demo", ["Enabled Wi-Fi"], "operational",
                       user_id="demo-user-a", thread_id="t1", status_profile="normal",
                       tickets_dir=tmp_path)
    saved = json.loads(next(tmp_path.glob("TKT-*.json")).read_text(encoding="utf-8"))["ticket"]
    assert r["status"] == "draft" and saved["ticket_id"] == r["ticket_id"]
    assert saved["steps_tried"] == ["Enabled Wi-Fi"]          # only the user's steps
    assert saved["user_id"] == "demo-user-a" and saved["thread_id"] == "t1"
    assert saved["reference_ids"] == ["GUIDE-WIFI"] and saved["status_profile"] == "normal"


def test_prepare_ticket_status_profile_only_when_checked(tmp_path):
    prepare_ticket("wifi", "x", ["y"], "not_checked", user_id="u", thread_id="t",
                   status_profile="normal", tickets_dir=tmp_path)
    saved = json.loads(next(tmp_path.glob("TKT-*.json")).read_text(encoding="utf-8"))["ticket"]
    assert saved["service_status"] == "not_checked" and saved["status_profile"] is None


def test_prepare_ticket_refuses_missing_details(tmp_path):
    r = prepare_ticket("wifi", "", [], user_id="u", thread_id="t", tickets_dir=tmp_path)
    assert r["error"] == "missing_details"
    assert set(r["missing"]) == {"problem description", "steps already tried"}
    assert list(tmp_path.glob("*.json")) == []                 # nothing written


def test_prepare_ticket_refuses_secrets(tmp_path):
    r = prepare_ticket("password", "forgot it", ["my password is hunter2"], user_id="u",
                       thread_id="t", tickets_dir=tmp_path)
    assert r["error"] == "secret_detected" and list(tmp_path.glob("*.json")) == []


def test_ticket_ids_are_unique(tmp_path):
    ids = {prepare_ticket("wifi", "x", ["y"], user_id="u", thread_id="t",
                          tickets_dir=tmp_path)["ticket_id"] for _ in range(5)}
    assert len(ids) == 5


def test_execute_tool_validates_arguments():
    ctx = {"status_profile": "normal", "user_id": "u", "thread_id": "t"}
    assert execute_tool("shell", {"cmd": "ls"}, ctx)["error"] == "unknown_tool"
    assert execute_tool("find_instructions", {"topic": "printer"}, ctx)["error"] == "invalid_arguments"
