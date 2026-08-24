# 🐧 linuxai

AI-powered Linux diagnostic & file-search CLI — with an agentic reasoning loop, three-layer safety review, and support for NVIDIA NIM, OpenRouter, or fully local Ollama models.

Instead of one shallow LLM call, **linuxai investigates problems step by step** — running real system commands, reading the results, deciding what to check next, running a three-layer safety review on every recommended fix, and verifying the fix worked. If it didn't, it re-plans and tries again.

---

## How it works

```
Your query
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  LangChain Orchestrator  (orchestrator.py)               │
│                                                          │
│  1. Classify: file search OR diagnostic?                 │
└──────────┬───────────────────────────┬───────────────────┘
           │ diagnostic                │ file search
           ▼                           ▼
┌──────────────────────┐   ┌────────────────────────────┐
│  Agentic loop ≤4     │   │  find_files() — one call   │
│  LLM picks tool      │   │  return results & exit     │
│  Whitelist check     │   └────────────────────────────┘
│  Param validation    │
│  Safe subprocess     │
│  Observe → memory    │
│  Repeat or break     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Diagnose            │
│  plain-English cause │
│  + candidate fixes   │
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────────────────────────┐
│  Three-layer safety review  (safety.py)                  │
│                                                          │
│  Layer 1 — generator tag  (diagnosis LLM's own label)    │
│  Layer 2 — deterministic pattern check  (no LLM)         │
│  Layer 3 — independent LLM review  (zero context)        │
│            → uses OLLAMA_SAFETY_MODEL if provider=ollama │
│                                                          │
│  Strictest wins: any "caution" → final tag = caution     │
└──────────┬───────────────────────────────────────────────┘
           ▼
┌──────────────────────┐
│  Human [y/N] gate    │  ← Python control flow, never bypassed
│  (cli.py)            │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Execute fix         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Verify              │
│  Before → After diff │
└──────────┬───────────┘
      ┌────┴────┐
   Resolved   Failed
      │           │
      ▼           ▼
    Done     Bounded retry ≤ 1
              (re-plan with failure
               in memory, new fix,
               full safety review,
               new confirmation)
```

---

## File structure

```
linuxai/
├── cli.py            # entrypoint — argparse, display, [y/N] gate, bounded retry
├── orchestrator.py   # LangChain LCEL workflow + explicit state management
├── safety.py         # three-layer safety classifier + dual-model Ollama support
├── agent.py          # classify → plan → diagnose (used by orchestrator)
├── gap_log.py        # local JSONL log of unsupported tool requests
├── tools.py          # whitelisted subprocess wrappers (11 tools)
├── llm_client.py     # OpenRouter / NVIDIA NIM / Ollama LLM wrapper
├── test_all.py       # 51 unit + integration tests (no API key needed)
├── .env              # API keys + provider config (never committed)
├── docker-compose.yml  # profiles: cloud (NVIDIA/OpenRouter) and ollama
├── docker/
│   ├── Dockerfile        # Ubuntu 22.04 + Python + all dependencies
│   └── seed_problems.sh  # seeds fake disk/memory issues for demo
└── README.md
```

---

## Quick start (local — macOS/Linux)

### 1. Install dependencies

```bash
pip3 install openai langchain langchain-core python-dotenv
```

### 2. Configure `.env`

Create a `.env` file in the project root — it is loaded automatically, no `export` needed:

```bash
# Choose provider: nvidia | openrouter | ollama
LINUXAI_PROVIDER=nvidia

# NVIDIA NIM
NVIDIA_API_KEY=nvapi-...

# OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Ollama (local — no key required)
# OLLAMA_MODEL=llama3
# OLLAMA_SAFETY_MODEL=mistral
```

### 3. Run

```bash
cd linuxai/
python3 cli.py "why is my disk full?"
python3 cli.py "memory usage is spiking"
python3 cli.py "what process is eating CPU?"
python3 cli.py "find all PDF files in /home"
python3 cli.py "is nginx running?"
python3 cli.py "check disk usage in /var/log"
python3 cli.py --provider openrouter "why is my disk full?"
```

### 4. Run tests (no API key needed)

```bash
python3 test_all.py          # plain output — 51 tests
python3 -m pytest test_all.py -v  # verbose with pytest
```

---

## LLM providers

Three providers — all use the **same `call_llm()` code path** in `llm_client.py`. Switch with one line in `.env`.

| Provider | Key needed? | Privacy | Default model |
|---|---|---|---|
| NVIDIA NIM | ✅ `NVIDIA_API_KEY` | Cloud | `meta/llama-3.1-8b-instruct` |
| OpenRouter | ✅ `OPENROUTER_API_KEY` | Cloud | `mistralai/mistral-7b-instruct:free` |
| Ollama | ❌ None — runs offline | 100% local | `llama3` |

### Two separate Ollama models

When using Ollama, linuxai uses **two different local models** for the two LLM call types:

```
Your query
    │
    ▼
classify / plan / diagnose  →  OLLAMA_MODEL        (e.g. llama3  — heavier, smarter)
    │
    ▼ (each recommended command)
independent safety review   →  OLLAMA_SAFETY_MODEL (e.g. mistral — lighter, faster)
                               falls back to OLLAMA_MODEL if not set
```

This lets a capable model do the reasoning while a fast, lightweight model handles the binary safe/caution classification.

### Using Ollama (local, private, no API key)

```bash
# 1. Install Ollama: https://ollama.com/download

# 2. Pull both models
ollama pull llama3    # main planning/diagnosis model
ollama pull mistral   # safety review model (lighter)

# 3. Set in .env
LINUXAI_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_SAFETY_MODEL=mistral

# 4. Run — fully offline
python3 cli.py "why is my disk full?"
```

---

## Docker demo (full Linux environment)

### Option A — Cloud providers (NVIDIA / OpenRouter)

```bash
cd linuxai/

# Start
docker compose --profile cloud up -d

# Enter interactive session
docker compose exec linuxai bash

# Inside:
seed_problems disk
linuxai "why is my disk full?"
```

### Option B — Ollama (local models, fully private)

```bash
# Start both containers (Ollama starts serving immediately)
docker compose --profile ollama up -d

# Pull models into the Ollama container (one-time, ~9 GB total)
docker exec ollama ollama pull llama3
docker exec ollama ollama pull mistral

# Enter linuxai
docker exec -it linuxai bash

# Inside — Ollama is reachable at http://ollama:11434 automatically:
export LINUXAI_PROVIDER=ollama
export OLLAMA_MODEL=llama3
export OLLAMA_SAFETY_MODEL=mistral

seed_problems disk
linuxai "why is my disk full?"
```

Models are stored in the `ollama_models` Docker volume — **no re-download on restart**.

### Seed problems & demo

```bash
# Inside the container:
seed_problems disk     # 500 MB fake log in /var/log (simulates disk bloat)
seed_problems memory   # stress-ng at 70% RAM for 600s (background)
seed_problems all      # disk + memory
seed_problems clean    # remove everything seeded

linuxai "why is my disk full?"
linuxai "memory usage is high"
linuxai "find all log files in /var"
linuxai "is ssh running?"
```

### Switch providers inside Docker

```bash
# NVIDIA
docker run -it \
  -e LINUXAI_PROVIDER=nvidia \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  linuxai bash

# OpenRouter
docker run -it \
  -e LINUXAI_PROVIDER=openrouter \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  linuxai bash
```

### `seed_problems` modes

| Command | What it does |
|---|---|
| `seed_problems disk` | 500 MB dummy log in `/var/log` |
| `seed_problems memory` | `stress-ng` at 70% RAM for 600 s (background) |
| `seed_problems all` | Disk bloat + memory pressure |
| `seed_problems clean` | Remove seeded file + kill stress-ng |

---

## Tool whitelist (11 tools — LLM cannot run anything outside this list)

### Zero-argument tools

| Tool | Command | Purpose |
|---|---|---|
| `check_disk` | `df -h` | Filesystem usage across all mounts |
| `check_dirs` | `du -sh /var/log /home` | Key directory sizes |
| `check_memory` | `free -h` | RAM + swap |
| `check_processes` | `ps aux` (sorted by mem) | Top 10 processes by memory |
| `check_logs` | `journalctl -p err` / `dmesg` | Last 20 error log entries |
| `check_open_ports` | `ss -tulwn` | Listening TCP/UDP ports |

### Parameterized tools (LLM supplies validated args)

| Tool | Args | Validation |
|---|---|---|
| `check_directory_size` | `path` | Must be under `/var/log`, `/home`, `/tmp`, `/var/cache`, `/var/lib`. Path traversal resolved and rejected. |
| `check_process_by_name` | `name` | Letters, digits, dash, underscore only. Filtering in Python, not shell. |
| `check_network` | `host` | RFC-1123 hostname or IPv4 only. `;`, `\|`, `&`, `` ` ``, `$`, spaces rejected. |
| `check_service_status` | `service_name` | Letters, digits, dash, underscore only. |
| `find_files` | `pattern`, `path` | Natural-language or glob. |

All parameterized tools use `subprocess.run([...], shell=False)` — **no shell injection is possible.**

---

## Three-layer safety review

Every fix command passes through three independent checks before the user sees it.

```
Candidate fix command
        │
        ▼
Layer 1 — Generator tag
  The diagnosis LLM's own self-assessment ("safe" / "caution")

        │
        ▼
Layer 2 — Deterministic pattern check  (safety.py — no LLM, always runs)
  Regex patterns that ALWAYS flag caution, regardless of LLM opinion:
  • rm -rf /          • dd of=/dev/*
  • mkfs              • chmod -R 777 /
  • fork bomb         • killall
  • systemctl stop/disable/mask

        │
        ▼
Layer 3 — Independent LLM safety review  (safety.py)
  A second LLM call with ZERO context about why the command was suggested.
  Sees only the command. Defaults to "caution" on any LLM error.
  When provider=ollama, uses OLLAMA_SAFETY_MODEL (lighter/faster model).

        │
        ▼
  Strictest wins:
  safe + safe + safe     →  safe
  safe + caution + safe  →  caution
  caution + safe + safe  →  caution
```

The reasoning is shown in the CLI next to every caution command:
```
[1] ⚠️  caution
    $ rm -f /var/log/fake_bloat.log
    ⚠ Why flagged: This command permanently deletes a file with no way to recover it.
```

---

## Bounded fix-retry

If a fix runs but verification shows the problem persists, linuxai re-plans — **once** (hard cap `MAX_FIX_RETRIES = 1`).

```
Fix approved → Execute → Verify
                              │
                    ┌─────────┴──────────┐
                 Resolved            Not resolved
                    │                    │
                   Done         Feed failure into memory
                                  + record failed command
                                  + increment retry count
                                         │
                              retry_count < MAX_FIX_RETRIES?
                                  ┌──────┴──────┐
                                 Yes             No
                                  │              │
                              Re-plan        "Manual investigation
                             New fix          may be required."
                           Safety review
                            Confirmation
                              Execute
                              Verify
```

The retry planner explicitly sees the failed command and is instructed not to recommend it again.

---

## Gap logging

When the LLM requests a tool not in the whitelist, it is rejected (never executed) and logged locally:

```json
{
  "timestamp": "2026-08-23T17:35:48Z",
  "query": "check postgres memory usage",
  "reason": "tool not in whitelist",
  "missing_tool": "check_postgres_memory",
  "partial_memory": [...]
}
```

The user sees:
```
⚠️  LLM requested unknown tool 'check_postgres_memory'. Treating as observation error.
📝 Capability gap recorded. No suitable diagnostic tool is currently available for this step.
```

Gap logs are **strictly local** — nothing is uploaded anywhere.

---

## LangChain orchestration

LangChain LCEL organises the workflow without replacing any underlying component.

```
orchestrator.py  (LangChain LCEL chain)
    │
    ├── step_classify       → file search OR diagnostic?
    ├── step_file_search    → fast path: find_files() and return
    ├── step_investigate    → agentic loop (whitelist → tools.py → memory)
    ├── step_diagnose       → final LLM diagnosis call
    └── step_safety_review  → three-layer review for all commands

WorkflowState (TypedDict):
  query, memory, diagnosis, commands, verification_tool,
  fix_command, failed_commands, fix_retry_count, resolved,
  provider, iterations_used, is_file_search, file_search_results
```

**LangChain cannot bypass the whitelist.** Every tool call goes through `tools.run_tool()`, which rejects any name not in `TOOL_REGISTRY` before touching subprocess.

---

## Ollama-specific reliability fixes

Running large prompts through local Ollama models with a small default context window (2048 tokens) caused `HTTP 500 unexpected EOF`. Three fixes are applied automatically when `LINUXAI_PROVIDER=ollama`:

| Fix | Where | What it does |
|---|---|---|
| Context window expansion | `_call_with_retry()` | Passes `num_ctx=4096` via `extra_body` — doubles the window |
| Memory truncation | `_memory_text()` | Each tool result capped at 500 chars before entering the prompt |
| Smaller response budget | `step_diagnose()` | `max_tokens` reduced 1024 → 512 for compact JSON output |

Cloud providers (NVIDIA, OpenRouter) are unaffected — these settings only activate when `provider == "ollama"`.

---

## Safety rules — enforced in code, not prompts

| Rule | Enforced in |
|---|---|
| Caution commands require `[y/N]` confirmation | `cli.py` — Python control flow |
| Tool names validated against `TOOL_REGISTRY` before execution | `tools.run_tool()` |
| Unknown tool → gap logged, error observation added, loop continues | `orchestrator.py` + `gap_log.py` |
| Parameterized args validated (path allowlist, regex, IP check) | `tools.py` — before subprocess |
| `shell=False` for all diagnostic tools | `tools.py` |
| `shell=True` only for user-approved fix commands | `cli.py` |
| Three-layer safety review on every fix command | `safety.py` |
| Deterministic pattern check cannot be overridden by any LLM | `safety.py` |
| Independent safety review defaults to "caution" on uncertainty | `safety.py` |
| Strictest safety layer always wins | `safety.finalize_safety_tag()` |
| Investigation loop hard-capped at 4 iterations | `orchestrator.py` |
| Fix-retry hard-capped at 1 | `orchestrator.py` `MAX_FIX_RETRIES = 1` |
| Gap logs are local only — no telemetry | `gap_log.py` |
| LangChain cannot bypass the whitelist | `orchestrator.py` → `tools.run_tool()` |
| Ollama context overflow prevented by num_ctx + memory truncation | `orchestrator.py` |

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LINUXAI_PROVIDER` | `openrouter` | `openrouter` / `nvidia` / `ollama` |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter |
| `NVIDIA_API_KEY` | — | Required for NVIDIA NIM |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct:free` | Model override |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | Model override |
| `OLLAMA_MODEL` | `llama3` | Main planning/diagnosis model |
| `OLLAMA_SAFETY_MODEL` | *(falls back to `OLLAMA_MODEL`)* | Lighter model for safety review only |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (auto-set to `http://ollama:11434` in Docker) |
| `LINUXAI_DEBUG` | — | Print full investigation memory trace |
| `NO_COLOR` | — | Disable ANSI colour output |

All variables are loaded automatically from a `.env` file in the project root — no `export` needed.
