"""Command line for Part 1 (no LLM, no API key).

Examples:
    python -m helpdesk.part1_cli --message "I cannot connect to Wi-Fi."
    python -m helpdesk.part1_cli --message "My Wi-Fi password does not work." --variant with_clarification
    python -m helpdesk.part1_cli --scenario S04        # writes outputs/S04.json
    python -m helpdesk.part1_cli --all                 # S01-S05
    python -m helpdesk.part1_cli --draw                # Mermaid drawings into outputs/
"""
from __future__ import annotations

import argparse
import json
import sys

from .data import OUTPUTS_DIR, get_scenario
from .part1 import build_graph, run_with_trace

PART1_SCENARIOS = ["S01", "S02", "S03", "S04", "S05"]


def print_trace(result: dict) -> None:
    print(f"message : {result['message']}")
    print("route   : START -> " + " -> ".join(result["nodes_visited"]) + " -> END")
    for t in result["trace"]:
        print(f"  [{t['node']}] update={json.dumps(t['update'], ensure_ascii=False)}")
        print(f"      state={json.dumps(t['state_after'], ensure_ascii=False)}")
    print("reply   :")
    print(result["final_reply"])


def run_scenario(scenario_id: str) -> dict:
    sc = get_scenario(scenario_id)
    variant = sc.get("graph_variant", "original")
    graph = build_graph(variant)
    events = []
    for ev in sc["events"]:
        if ev["action"] != "message":
            continue
        res = run_with_trace(graph, ev["message"])
        res["user_id"], res["thread_id"] = ev["user_id"], ev["thread_id"]
        res["information_source"] = (
            {"reference_file": "data/helpdesk.json"}
            if res["nodes_visited"][-1] in ("wifi_help", "password_help")
            else {"reference_file": None, "note": "fixed question, no lookup"}
        )
        events.append(res)
    log = {
        "scenario": sc["id"],
        "title": sc["title"],
        "command": f"python -m helpdesk.part1_cli --scenario {sc['id']}",
        "graph_variant": variant,
        "status_profile": sc["status_profile"],
        "model_used": None,
        "events": events,
    }
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / f"{sc['id']}.json"
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"=== {sc['id']} - {sc['title']} (graph: {variant}) -> {out.name}")
    for r in events:
        print_trace(r)
    print()
    return log


def draw() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    for variant, name in (("original", "graph_part1_original.mmd"),
                          ("with_clarification", "graph_part1_clarify.mmd")):
        mermaid = build_graph(variant).get_graph().draw_mermaid()
        (OUTPUTS_DIR / name).write_text(mermaid, encoding="utf-8")
        print(f"wrote outputs/{name}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="HW5 Part 1: helpdesk graph without an LLM")
    p.add_argument("--message")
    p.add_argument("--variant", choices=["original", "with_clarification"], default="original")
    p.add_argument("--scenario")
    p.add_argument("--all", action="store_true")
    p.add_argument("--draw", action="store_true")
    a = p.parse_args()
    if a.draw:
        draw()
    if a.all:
        for sid in PART1_SCENARIOS:
            run_scenario(sid)
    elif a.scenario:
        run_scenario(a.scenario)
    elif a.message:
        print_trace(run_with_trace(build_graph(a.variant), a.message))
    elif not a.draw:
        p.print_help()


if __name__ == "__main__":
    main()
