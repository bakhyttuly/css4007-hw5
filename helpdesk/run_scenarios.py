"""Run supplied scenarios against the full helpdesk (model + tools + memory).

    python -m helpdesk.run_scenarios --part2            # S06-S09, S14
    python -m helpdesk.run_scenarios --part3            # S10-S13
    python -m helpdesk.run_scenarios S11                # any scenario by id
    python -m helpdesk.run_scenarios S11 --fake         # offline test double (logs -> outputs/fake-runs/)

Memory lives in HELPDESK_STORAGE_DIR (default local-storage/). Before each scenario
the storage is deleted (reset_before). A `restart_process` event really ends the
Python process: every segment of a scenario that has restarts runs in its own
`python` subprocess against the same storage files.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .agent import build_agent, make_config, run_turn
from .data import OUTPUTS_DIR, get_scenario
from .models import get_model
from .storage import open_storage, reset_storage, storage_dir

PART2 = ["S06", "S07", "S08", "S09", "S14"]
PART3 = ["S10", "S11", "S12", "S13"]


def split_segments(events: list[dict]) -> list[list[dict]]:
    segments, cur = [], []
    for ev in events:
        if ev["action"] == "restart_process":
            segments.append(cur)
            cur = []
        elif ev["action"] == "message":
            cur.append(ev)
    segments.append(cur)
    return segments


def run_segment(sid: str, index: int, fake: bool, storage: Path, out_dir: Path) -> list[dict]:
    """Run one segment (one process lifetime) of a scenario."""
    sc = get_scenario(sid)
    segment = split_segments(sc["events"])[index]
    model, _ = get_model(fake)
    results = []
    with open_storage(storage) as (checkpointer, store):
        agent = build_agent(model, checkpointer=checkpointer, store=store,
                            tickets_dir=out_dir / "tickets")
        for ev in segment:
            cfg = make_config(ev["user_id"], ev["thread_id"], sc["status_profile"])
            res = run_turn(agent, ev["message"], cfg)
            mem = res["user_memory_read"]
            res.update(user_id=ev["user_id"], thread_id=ev["thread_id"], pid=os.getpid())
            res["information_sources"] = {
                "reference_file": "data/helpdesk.json" if "find_instructions" in res["tools_used"] else None,
                "status_snapshot": sc["status_profile"] if "check_service_status" in res["tools_used"] else None,
                "conversation_state": "checkpointer, thread_id=" + ev["thread_id"],
                "user_memory": mem or "nothing saved for this user",
            }
            results.append(res)
    return results


def run_scenario(sid: str, fake: bool, storage: Path, out_dir: Path) -> dict:
    sc = get_scenario(sid)
    _, info = get_model(fake)
    if sc.get("reset_before", True):
        reset_storage(storage)
    segments = split_segments(sc["events"])
    events: list[dict] = []
    flags = (["--fake"] if fake else [])
    for i in range(len(segments)):
        if len(segments) == 1:
            events += run_segment(sid, i, fake, storage, out_dir)
            continue
        with tempfile.TemporaryDirectory() as tmp:
            result_file = Path(tmp) / "r.json"
            subprocess.run([sys.executable, "-m", "helpdesk.run_scenarios", "--segment", sid, str(i),
                            "--storage-dir", str(storage), "--out-dir", str(out_dir),
                            "--result-file", str(result_file)] + flags, check=True)
            events += json.loads(result_file.read_text(encoding="utf-8"))
    cmd = f"python -m helpdesk.run_scenarios {sid}" + (" --fake" if fake else "")
    log = {"scenario": sid, "title": sc["title"], "command": cmd,
           "status_profile": sc["status_profile"], "model": info,
           "process_restarts": len(segments) - 1,
           "process_ids": sorted({e["pid"] for e in events}), "events": events}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{sid}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(f"=== {sid} - {sc['title']}  [{info['provider']}: {info['model']}]  snapshot={sc['status_profile']}"
          f"  restarts={len(segments) - 1}")
    for r in events:
        print(f"[pid {r['pid']}] user={r['user_id']} thread={r['thread_id']}")
        print(f"user : {r['message']}")
        print(f"nodes: {' -> '.join(r['nodes_visited'])}")
        print(f"tools: {r['tools_used']}  user_memory_read={r['user_memory_read']}  "
              f"loop_limit_reached={r['loop_limit_reached']}")
        print(f"reply: {r['final_reply']}")
        if r["tickets"]:
            print(f"ticket file: {r['tickets']}")
        print()
    return log


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("scenarios", nargs="*")
    p.add_argument("--part2", action="store_true")
    p.add_argument("--part3", action="store_true")
    p.add_argument("--fake", action="store_true", help="offline test double, NOT a real model")
    p.add_argument("--storage-dir")
    p.add_argument("--out-dir")
    p.add_argument("--segment", nargs=2, metavar=("SCENARIO", "INDEX"), help=argparse.SUPPRESS)
    p.add_argument("--result-file", help=argparse.SUPPRESS)
    a = p.parse_args()
    storage = storage_dir(a.storage_dir)
    out_dir = Path(a.out_dir) if a.out_dir else (OUTPUTS_DIR / "fake-runs" if a.fake else OUTPUTS_DIR)

    if a.segment:  # internal: one process lifetime of a scenario
        res = run_segment(a.segment[0].upper(), int(a.segment[1]), a.fake, storage, out_dir)
        Path(a.result_file).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return

    ids = [s.upper() for s in a.scenarios] + (PART2 if a.part2 else []) + (PART3 if a.part3 else [])
    if not ids:
        p.print_help()
        return
    for sid in ids:
        run_scenario(sid, a.fake, storage, out_dir)


if __name__ == "__main__":
    main()
