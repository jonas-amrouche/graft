# Graft

**A local-first AI coding tool built around a design language called Mid.**

Mid Is Design

Instead of prompting AI directly for code, you build a *Mid document* — a plain-English spec in a structured format — and the AI generates code from that. The Mid document lives in your repo, evolves with your project, and gives you granular control over what the AI can and cannot change.

---

## The idea

Most AI coding tools let the AI write whatever it wants. Graft adds a layer between your intent and generated code: the **Mid document**.

```
Your prompt → Mid document → Generated code
```

The Mid document is the source of truth. It describes your app in six structured section types. Each section type carries a different level of human vs AI authority. The AI generates code from the Mid, and generated code is tagged back to the sections that produced it — so you can regenerate one section's code without touching the rest.

---

## Mid — the design language

Every app in Graft is described as a set of **parts** (domains) containing **sections** (one idea each). Sections have six types:

| Keyword | What it describes | Authority |
|---|---|---|
| `ANCHOR` | Invariants — things that must always be true | Human-only, always locked |
| `STRUCTURE` | UI layout and screen regions | Human-approved, frozen after approval |
| `DATA` | Data shapes and in-memory state | Collaborative |
| `WHEN` | One user action and its exact result | AI workbench |
| `ASSERT` | Validation rules and constraints | Human-confirmed |
| `SURFACE` | All visible text — labels, messages, empty states | AI-owned |

A Mid file looks like this:

```
PART: Tasks
NAME: Tasks
STATUS: draft

DATA: Task List
MARKER: task_list
INTENT: holds all tasks and their state
---
The app holds a list of tasks. Each task has a title, a completion status, and a creation date.

STRUCTURE: Task Screen
MARKER: task_screen
INTENT: the main screen showing all tasks
---
The screen has a top bar with the app name, a scrollable list of task cards, and a floating add button in the bottom right.

WHEN: Complete Task
MARKER: complete_task
INTENT: user marks a task as done
---
When the user taps the checkbox on a task card, the task status changes to complete. The card moves to the bottom of the list with a strikethrough style.
```

Mid files are stored as `.mid` in a `projects/` folder. They are plain text and readable as documentation.

---

## Features

- **Mid editor** — write prompts, watch the Mid document build up section by section
- **Code generation** — compile the full Mid to a working single-file HTML/JS/CSS app
- **Surgical recompile** — regenerate one section's code block without touching the rest
- **Code ownership** — generated code is tagged with `// [graft:marker]` regions tied back to Mid sections; ANCHOR and locked STRUCTURE regions are protected and restored if the AI changes them
- **Section locking** — approve or lock sections so the AI cannot modify them
- **Undo/redo** — full snapshot history, keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z)
- **Prompt history** — every prompt logged with what changed
- **Part verification** — snapshot a part; the AI is warned not to contradict it
- **Rich body text** — cross-references, syntax tokens, noun phrase detection
- **Code view** — region labels that link back to their Mid section
- **Two AI backends** — Ollama (local) or Anthropic API (cloud), switchable per session

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (for local mode)
- An Anthropic API key (for cloud mode — optional)

---

## Setup

```bash
git clone https://github.com/jonas-amrouche/graft.git
cd graft
chmod +x start.sh
./start.sh
```

`start.sh` creates a virtualenv, installs dependencies, and opens `http://localhost:8000`.

**Manual setup:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## Recommended models

Graft uses two separate model roles — one for compiling prompts into Mid, one for generating code from Mid.

**Local (Ollama):**

| Role | Recommended model | Pull command |
|---|---|---|
| Intent compiler (Mid) | `mistral:latest` | `ollama pull mistral` |
| Code compiler | `qwen2.5-coder:3b` | `ollama pull qwen2.5-coder:3b` |

For better Mid quality locally, try `qwen2.5:14b` or `llama3.1:8b` as the intent compiler.

**Cloud (Anthropic):**

Both roles default to `claude-sonnet-4-20250514`. Enter your API key in the settings panel (gear icon). The key is stored only in `graft_config.json` on your machine and is never sent anywhere except the Anthropic API.

---

## Configuration

Settings are stored in `graft_config.json` (created on first run, gitignored):

```json
{
  "intent_compiler": {
    "source": "ollama",
    "ollama_model": "mistral:latest",
    "ollama_url": "http://localhost:11434",
    "anthropic_model": "claude-sonnet-4-20250514",
    "anthropic_key": ""
  },
  "code_compiler": {
    "source": "ollama",
    "ollama_model": "qwen2.5-coder:3b",
    "ollama_url": "http://localhost:11434",
    "anthropic_model": "claude-sonnet-4-20250514",
    "anthropic_key": ""
  },
  "mid_verbosity": "moderate"
}
```

---

## Project files

Each project stores files in `projects/`:

```
projects/
  my_app.mid              # Mid document — commit this
  my_app.parts.json       # Part metadata (status, snapshots) — commit this
  my_app.history.json     # Prompt history — optional
  my_app.ownership.json   # Code region map — optional
  my_app_output/
    my_app.html           # Generated code
```

---

## Status

Early experiment. The core loop (prompt → Mid → code → surgical recompile) works. Rough edges exist, especially around AI marker consistency in generated code. No tests yet.

Contributions welcome.

---

## License

MIT
