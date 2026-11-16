# CSS-4007 HW5 - University IT helpdesk with LangGraph

A terminal helpdesk assistant built in four stages with LangGraph. All data is fictional
(`data/`); tickets are **local drafts only**, nothing is ever sent. The original task text is
in [ASSIGNMENT.md](ASSIGNMENT.md); my explanations, predictions and results are in
[SUBMISSION.md](SUBMISSION.md).

| Part | What it is | Needs API key |
| --- | --- | --- |
| 1 | Graph with a conditional edge, no LLM (`helpdesk/part1.py`) | no |
| 2 | Model chooses tools in a ReAct loop with a limit (`helpdesk/agent.py`, `helpdesk/tools.py`) | yes (real runs) |
| 3 | Conversation memory (thread) + user memory (user), SQLite | yes (real runs) |
| 4 | Tests + GitHub Actions + this README | no |

## Versions

* Python **3.12.2** (CI uses Python 3.12).
* Exact package versions are pinned in [requirements.txt](requirements.txt)
  (LangGraph 1.2.22, langgraph-checkpoint-sqlite 3.1.3, langchain-openai 1.8.0).
* Real-model runs: provider **OpenAI**, model **gpt-4o-mini**, temperature 0.

## Install

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Configure the model (Parts 2-3 with a real model)

```text
copy .env.example .env        # Windows   (macOS/Linux: cp .env.example .env)
```

Edit `.env`:

```text
OPENAI_API_KEY=<your key>
MODEL_NAME=gpt-4o-mini
HELPDESK_STORAGE_DIR=local-storage
```

`.env` is ignored by git. **Never commit a key.** Another provider can be used by changing
`helpdesk/models.py` (`get_real_model`) and the adapter package.

## Run

**Part 1 (no key):**

```text
python -m helpdesk.part1_cli --message "I cannot connect to Wi-Fi."
python -m helpdesk.part1_cli --message "My Wi-Fi password does not work." --variant with_clarification
python -m helpdesk.part1_cli --all        # S01-S05, logs in outputs/S01.json ... S05.json
python -m helpdesk.part1_cli --draw       # Mermaid drawings of both graphs into outputs/
```

**Interactive helpdesk (model + tools + memory):**

```text
python -m helpdesk.chat --user demo-user-a --thread t1 --profile normal
```

* `--user` is the user identifier (owner of long-term preferences).
* `--thread` is the conversation identifier. Start the command again with the same `--user` and
  `--thread` to resume that conversation after closing the program.
* `--profile` selects the service-status snapshot: `normal`, `wifi_outage`, `recovery_outage`,
  `status_unavailable`.

**Supplied scenarios S01-S14 (real model):**

```text
python -m helpdesk.part1_cli --all                 # S01-S05
python -m helpdesk.run_scenarios --part2           # S06, S07, S08, S09, S14
python -m helpdesk.run_scenarios --part3           # S10, S11, S12, S13
python -m helpdesk.run_scenarios S11               # one scenario
```

Logs are written to `outputs/<ID>.json` (nodes visited, model tool requests, tool results, user
memory read, final reply, ticket filename, whether the loop limit was reached, process ids).
Scenarios with a `restart_process` event run each part in a **separate Python process** against
the same storage files; the `process_ids` field in the log shows this.

The same runner with `--fake` uses the offline test double and writes to `outputs/fake-runs/`.
Those logs are **not** real-model results.

## Where memory is stored and how to clear it

| Memory | LangGraph component | Key | File |
| --- | --- | --- | --- |
| Conversation (messages, steps tried) | SQLite checkpointer | `thread_id` | `local-storage/checkpoints.sqlite` |
| User preferences (reply language) | SQLite store, namespace `("preferences", user_id)` | `user_id` | `local-storage/user_memory.sqlite` |

`local-storage/` is git-ignored (so are `*.sqlite`). Ticket drafts are JSON files in
`outputs/tickets/` (invented demo data only).

Clear all saved memory and start again:

```text
python -m helpdesk.storage --reset
```

(or delete the `local-storage` folder). The scenario runner does this automatically before
each scenario (`reset_before`).

## Tests

```text
python -m pytest -q
```

53 tests, no API key. Model behaviour is replaced by `helpdesk/fake_model.py`, a small
rule-based double that is **labelled as not an LLM** in its name, its docstring and in every
log it produces.

| File | What it checks | What it would catch if the code broke |
| --- | --- | --- |
| `tests/test_routing.py` | The four Part 1 messages take the predicted route; matching ignores case; mixed Wi-Fi + password goes to `wifi_help` in the original graph and to `clarify_problem` in the changed graph; help text comes from `helpdesk.json` | A changed keyword rule, case-sensitive matching, the wrong rule order, a broken clarification branch, help text no longer read from the data file |
| `tests/test_tools.py` | Each tool called directly: instructions lookup, all four status snapshots, ticket draft contents/uniqueness, refusal of missing details and of secrets, argument validation | Tools that only work through the agent, a status read from the wrong snapshot, tickets with invented fields, tickets saved without the user's steps, passwords written into tickets |
| `tests/test_agent.py` | The conditional ticket request behaves differently for operating / outage / unknown; password problems check `account_recovery`; missing details lead to a question; the model cannot pick the status snapshot; the loop limit stops a model that never stops and the budget resets each turn; code blocks conditional or unrequested tickets even if the model ignores the rule; the ticket records the real status | A loop that can run forever, always creating a ticket, the model overriding trusted context, a ticket created in an outage or when the user never asked, a budget that is not reset between messages |
| `tests/test_memory.py` | Preference saved per user; a conversation survives closing and reopening the storage (and a real process restart); a new thread of the same user has the preference but not the old conversation; another user sees nothing; the model cannot choose whose memory is written; reset clears everything; invalid language values are rejected | Memory kept only in RAM, thread and user memory mixed up, one user's preference leaking to another, a hard-coded preference, reset that does not clear |

## CI

[.github/workflows/tests.yml](.github/workflows/tests.yml) runs `pytest` on every push and pull
request (Ubuntu, Python 3.12, pinned requirements). It needs no secrets and uses the fake model.
The real-model runs of Parts 2-3 are made by hand and saved in `outputs/`.

## Project layout

```text
helpdesk/part1.py        Part 1 graph (original + with_clarification), tracing
helpdesk/part1_cli.py    Part 1 command line / scenarios S01-S05
helpdesk/tools.py        find_instructions, check_service_status, prepare_ticket, remember_language_preference
helpdesk/agent.py        model + tools graph, loop limit, memory loading
helpdesk/storage.py      SQLite checkpointer + store, reset
helpdesk/memory.py       user preference store helpers
helpdesk/models.py       real model setup (OpenAI) / fake model switch
helpdesk/fake_model.py   offline test double (NOT an LLM)
helpdesk/chat.py         interactive terminal chat
helpdesk/run_scenarios.py scenario runner with real process restarts
data/                    supplied fictional fixtures
outputs/                 saved logs and ticket drafts
tests/                   pytest suite
```

## Privacy

Only invented identifiers (`demo-user-a`, `demo-user-b`) are used. Keys live in `.env`
(ignored). Memory databases live in `local-storage/` (ignored). Tickets must not contain
passwords or recovery codes; `prepare_ticket` rejects them.
