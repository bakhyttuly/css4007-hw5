"""Part 3 tests: conversation memory (thread_id) vs user memory (user_id), with restarts.

They use the labelled FAKE model (no API key). Storage is a temporary directory.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from helpdesk.agent import build_agent, make_config, run_turn
from helpdesk.fake_model import FakeRuleModel
from helpdesk.memory import load_user_memory, save_language
from helpdesk.storage import open_storage

REPO = Path(__file__).resolve().parent.parent


def session(storage, user, thread, text, profile="normal"):
    """One 'process lifetime': open the files, send one message, close the files."""
    with open_storage(storage) as (cp, store):
        agent = build_agent(FakeRuleModel(), checkpointer=cp, store=store,
                            tickets_dir=storage / "tickets")
        return run_turn(agent, text, make_config(user, thread, profile))


def test_save_and_load_language_per_user(tmp_path):
    with open_storage(tmp_path) as (_, store):
        assert save_language(store, "demo-user-a", "kazakh")["value"] == "Kazakh"
        assert load_user_memory(store, "demo-user-a") == {"language": "Kazakh"}
        assert load_user_memory(store, "demo-user-b") == {}


@pytest.mark.parametrize("bad", ["", "x", "Kazakh; DROP TABLE", "12345", "a" * 80])
def test_invalid_language_is_rejected(tmp_path, bad):
    with open_storage(tmp_path) as (_, store):
        assert save_language(store, "demo-user-a", bad)["error"] == "invalid_language"
        assert load_user_memory(store, "demo-user-a") == {}


def test_nothing_is_preloaded_for_a_new_user(tmp_path):
    r = session(tmp_path, "demo-user-a", "t1", "Which language did I ask you to use?")
    assert "No preference" in r["final_reply"] and r["user_memory_read"] == {}


def test_conversation_survives_restart(tmp_path):                      # S11
    session(tmp_path, "demo-user-a", "restart-a",
            "My issue is Wi-Fi. I enabled it and tried reconnecting, but it still fails.")
    r = session(tmp_path, "demo-user-a", "restart-a", "Which problem and steps did we discuss?")
    assert "Enabled Wi-Fi" in r["final_reply"] and "Tried reconnecting" in r["final_reply"]


def test_preference_is_saved_then_found_in_new_thread_after_restart(tmp_path):   # S12
    r1 = session(tmp_path, "demo-user-a", "preference-a1",
                 "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    assert r1["tools_used"] == ["remember_language_preference"]
    r2 = session(tmp_path, "demo-user-a", "preference-a2", "Which language did I ask you to use?")
    assert "Kazakh" in r2["final_reply"] and r2["user_memory_read"] == {"language": "Kazakh"}
    # the old conversation is NOT visible in the new thread
    r3 = session(tmp_path, "demo-user-a", "preference-a2", "Which problem did we discuss?")
    assert "no earlier messages" in r3["final_reply"]


def test_other_user_does_not_see_the_preference(tmp_path):             # S13
    session(tmp_path, "demo-user-a", "isolation-a",
            "Please remember that I prefer replies in Kazakh. I have a Wi-Fi problem.")
    r = session(tmp_path, "demo-user-b", "isolation-b", "Have I told you a preferred language?")
    assert "Kazakh" not in r["final_reply"] and r["user_memory_read"] == {}
    assert "No preference" in r["final_reply"]


def test_same_thread_id_for_different_users_is_still_isolated_for_preferences(tmp_path):
    session(tmp_path, "demo-user-a", "shared", "Please remember that I prefer replies in Kazakh.")
    r = session(tmp_path, "demo-user-b", "other", "Which language did I ask you to use?")
    assert "Kazakh" not in r["final_reply"]


def test_model_cannot_pick_whose_memory_is_written(tmp_path):
    from helpdesk.tools import execute_tool
    with open_storage(tmp_path) as (_, store):
        ctx = {"user_id": "demo-user-a", "thread_id": "t", "status_profile": "normal", "store": store}
        execute_tool("remember_language_preference",
                     {"language": "Kazakh", "user_id": "demo-user-b"}, ctx)
        assert load_user_memory(store, "demo-user-a") == {"language": "Kazakh"}
        assert load_user_memory(store, "demo-user-b") == {}


def test_storage_reset_clears_everything(tmp_path):
    from helpdesk.storage import reset_storage
    session(tmp_path / "mem", "demo-user-a", "t", "Please remember that I prefer replies in Kazakh.")
    reset_storage(tmp_path / "mem")
    r = session(tmp_path / "mem", "demo-user-a", "t", "Which language did I ask you to use?")
    assert r["user_memory_read"] == {}


def test_real_process_restart_with_the_scenario_runner(tmp_path):      # S11 with a real restart
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "helpdesk.run_scenarios", "S11", "--fake",
                    "--storage-dir", str(tmp_path / "mem"), "--out-dir", str(out)],
                   cwd=REPO, check=True, capture_output=True)
    log = json.loads((out / "S11.json").read_text(encoding="utf-8"))
    assert log["process_restarts"] == 1 and len(log["process_ids"]) == 2
    assert "Enabled Wi-Fi" in log["events"][-1]["final_reply"]
