"""Persistent storage (Part 3): two SQLite files in HELPDESK_STORAGE_DIR.

  checkpoints.sqlite  - LangGraph checkpointer: conversation state, keyed by thread_id
  user_memory.sqlite  - LangGraph store: user preferences, keyed by user_id

Both live in an ignored directory (default: local-storage/). Reset them with
    python -m helpdesk.storage --reset
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

REPO_ROOT = Path(__file__).resolve().parent.parent


def storage_dir(path: str | Path | None = None) -> Path:
    raw = path or os.getenv("HELPDESK_STORAGE_DIR") or "local-storage"
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def reset_storage(path: str | Path | None = None) -> Path:
    d = storage_dir(path)
    if d.exists():
        shutil.rmtree(d)
    return d


@contextmanager
def open_storage(path: str | Path | None = None):
    """Yield (checkpointer, store). Closing the context closes both database files."""
    d = storage_dir(path)
    d.mkdir(parents=True, exist_ok=True)
    cp_conn = sqlite3.connect(d / "checkpoints.sqlite", check_same_thread=False)
    st_conn = sqlite3.connect(d / "user_memory.sqlite", check_same_thread=False,
                              isolation_level=None)
    try:
        checkpointer = SqliteSaver(cp_conn)
        checkpointer.setup()
        store = SqliteStore(st_conn)
        store.setup()
        yield checkpointer, store
    finally:
        cp_conn.close()
        st_conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Manage the local memory storage")
    ap.add_argument("--reset", action="store_true", help="delete all saved memory")
    ap.add_argument("--dir")
    a = ap.parse_args()
    if a.reset:
        print(f"deleted {reset_storage(a.dir)}")
    else:
        print(f"storage directory: {storage_dir(a.dir)}")
