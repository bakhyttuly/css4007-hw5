"""Interactive terminal chat with persistent memory.

    python -m helpdesk.chat --user demo-user-a --thread t1 --profile normal
    python -m helpdesk.chat --user demo-user-a --thread t1      # run again later: same thread resumes
    python -m helpdesk.chat --fake                              # offline test double

Memory is stored in HELPDESK_STORAGE_DIR (default local-storage/). Clear it with
    python -m helpdesk.storage --reset
"""
from __future__ import annotations

import argparse
import sys

from .agent import build_agent, make_config, run_turn
from .data import load_status_profiles
from .models import get_model
from .storage import open_storage, storage_dir


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--user", default="demo-user-a")
    p.add_argument("--thread", default="chat-1")
    p.add_argument("--profile", default="normal", choices=list(load_status_profiles()))
    p.add_argument("--fake", action="store_true")
    p.add_argument("--storage-dir")
    a = p.parse_args()
    model, info = get_model(a.fake)
    with open_storage(a.storage_dir) as (checkpointer, store):
        agent = build_agent(model, checkpointer=checkpointer, store=store)
        cfg = make_config(a.user, a.thread, a.profile)
        print(f"[{info['provider']}: {info['model']}] user={a.user} thread={a.thread} "
              f"status={a.profile} storage={storage_dir(a.storage_dir)}. Type 'exit' to quit.")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("exit", "quit"):
                break
            if text:
                r = run_turn(agent, text, cfg)
                print(f"  (nodes: {' -> '.join(r['nodes_visited'])}; tools: {r['tools_used']})")
                print(f"bot> {r['final_reply']}")


if __name__ == "__main__":
    main()
