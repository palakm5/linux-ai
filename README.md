# 🐧 linuxai

AI-powered Linux diagnostic & file-search CLI.

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
│  Layer 1 — generator tag (diagnosis LLM's own label)     │
│  Layer 2 — deterministic pattern check  (no LLM)         │
│  Layer 3 — independent LLM review  (zero context)        │
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
├── agent.py          # classify → plan → diagnose (used by orchestrator)
├── orchestrator.py   # LangChain LCEL workflow + explicit state management
├── safety.py         # three-layer safety classifier
├── gap_log.py        # local JSONL log of unsupported tool requests
├── tools.py          # whitelisted subprocess wrappers (11 tools)
├── llm_client.py     # OpenRouter + NVIDIA NIM dual-path LLM wrapper
├── docker/
│   ├── Dockerfile        # Ubuntu 22.04 + Python + linuxai
│   └── seed_problems.sh  # seeds fake disk/memory issues for demo
└── README.md
```

---

## Quick start (local — macOS/Linux)

### 1. Install dependencies

```bash
pip3 install openai langchain langchain-core
```

### 2. Set your API key

**OpenRouter** (free tier available — recommended):
```bash
export OPENROUTER_API_KEY=sk-or-...
```

**NVIDIA NIM** (alternative):
```bash
export NVIDIA_API_KEY=nvapi-...
export LINUXAI_PROVIDER=nvidia
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
python3 cli.py --provider nvidia "why is my disk full?"
```

---

## Docker demo (full Linux environment)

### Install Docker Desktop (macOS)

1. Download from **https://www.docker.com/products/docker-desktop/**
2. Open the `.dmg`, drag Docker to Applications, launch it
3. Wait for the whale icon to stop animating (~30 s)
4. Verify: `docker --version`

### Build the image

```bash
cd linuxai/
docker build -f docker/Dockerfile -t linuxai .
```

### Run the container

```bash
docker run -it \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  linuxai
```

You'll get a bash shell inside Ubuntu 22.04.

### Seed problems & demo

```bash
# Inside the container:

# Seed a 500 MB fake log file (simulates disk bloat)
seed_problems disk

# Seed memory pressure (stress-ng at 70% RAM, background)
seed_problems memory

# Seed both at once
seed_problems

# Run linuxai — it investigates step by step, then asks before running fixes
linuxai "why is my disk full?"
linuxai "my system feels slow"
linuxai "is nginx running?"
linuxai "find all log files in /var"

# The CLI shows a before/after comparison after each approved fix.
# If the fix didn't work, it re-plans automatically (bounded to 1 retry).

# Clean up seeded files
seed_problems clean
```

### `seed_problems` modes

| Command | What it does |
|---|---|
| `seed_problems` | Disk bloat + memory pressure |
| `seed_problems disk` | 500 MB dummy log in `/var/log` |
| `seed_problems memory` | `stress-ng` at 70% RAM for 600s, background |
| `seed_problems clean` | Remove seeded file + kill stress-ng |

### Switch providers inside the container

```bash
docker run -it \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e LINUXAI_PROVIDER=nvidia \
  linuxai
```

---

## Tool whitelist (11 tools — LLM cannot run anything outside this list)

### Zero-argument tools

| Tool | Command | Purpose |
|---|---|---|
| `check_disk` | `df -h` | Filesystem usage across all mounts |
| `check_dirs` | `du -sh /var/log /home` | Key directory sizes |
| `check_memory` | `free -h` | RAM + swap |
| `check_processes` | `ps aux --sort=-%mem` | Top 10 processes by memory |
| `check_logs` | `journalctl -p err` / `dmesg` | Last 20 error log entries |
| `check_open_ports` | `ss -tulwn` | Listening TCP/UDP ports |

### Parameterized tools (LLM supplies validated args)

| Tool | Args | Validation | Command |
|---|---|---|---|
| `check_directory_size` | `path` | Must be under `/var/log`, `/home`, `/tmp`, `/var/cache`, or `/var/lib`. Path traversal (`../../`) is resolved and rejected. | `du -sh <path>` |
| `check_process_by_name` | `name` | Letters, digits, dash, underscore only. Shell metacharacters rejected. Filtering done in Python, not shell. | `ps aux` + Python filter |
| `check_network` | `host` | RFC-1123 hostname or IPv4 only. `;`, `\|`, `&`, `` ` ``, `$`, spaces rejected. | `ping -c 4 <host>` |
| `check_service_status` | `service_name` | Letters, digits, dash, underscore only. | `systemctl status <name>` |
| `find_files` | `pattern`, `path` | Natural-language or glob. Time keywords resolved. | `find <path> -iname <glob>` |

All parameterized tools use `subprocess.run([...], shell=False)` — **no shell injection is possible**.

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
Layer 2 — Deterministic pattern check  (safety.py — no LLM)
  Regex patterns that ALWAYS flag caution, regardless of LLM opinion:
  • rm -rf /          • dd of=/dev/*
  • mkfs              • chmod -R 777 /
  • fork bomb         • killall
  • systemctl stop/disable/mask

        │
        ▼
Layer 3 — Independent LLM safety review  (safety.py)
  A second LLM call with ZERO context about why the command was suggested.
  Sees only the command. Defaults to "caution" when uncertain.

        │
        ▼
  Strictest wins:
  safe + safe + safe  →  safe
  safe + caution + safe  →  caution
  caution + safe + safe  →  caution
```

The reasoning is shown in the CLI next to every caution command:
```
[CAUTION] rm -f /var/log/fake_bloat.log
  ⚠ Why flagged: This command permanently deletes a file with no way to recover it.
```

---

## Bounded fix-retry

If a fix runs but verification shows the problem persists, linuxai automatically re-plans — **once** (hard cap `MAX_FIX_RETRIES = 1`).

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

When the LLM requests a tool that doesn't exist in the whitelist, the request is rejected (never executed) and logged locally to `tool_gap_log.jsonl`:

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

LangChain organises the workflow using LCEL (LangChain Expression Language). It never replaces the underlying components — it coordinates them.

```
orchestrator.py  (LangChain LCEL chain)
    │
    ├── step_classify     → classify query type
    ├── step_investigate  → agentic loop (calls tools.py via whitelist)
    ├── step_diagnose     → final LLM diagnosis call
    └── step_safety_review → three-layer review for all commands

Explicit workflow state (WorkflowState TypedDict):
  query, memory, diagnosis, commands, verification_tool,
  fix_command, failed_commands, fix_retry_count, resolved,
  provider, iterations_used, is_file_search, file_search_results
```

**LangChain cannot bypass the whitelist.** Every tool call goes through `tools.run_tool()`, which rejects any name not in `TOOL_REGISTRY` before touching subprocess.

---

## Safety rules — enforced in code, not prompts

| Rule | Enforced in |
|---|---|
| Caution commands require `[y/N]` confirmation | `cli.py` — Python control flow |
| Tool names validated against `TOOL_REGISTRY` before execution | `tools.run_tool()` |
| Unknown tool → gap logged, error observation added, loop continues | `agent.py` + `gap_log.py` |
| Parameterized args validated (path allowlist, regex, IP check) | `tools.py` — before subprocess |
| `shell=False` (default) for all diagnostic tools | `tools.py` |
| `shell=True` only for user-approved caution commands | `cli.py` |
| Three-layer safety review on every fix command | `safety.py` |
| Deterministic pattern check cannot be overridden by any LLM | `safety.py` |
| Independent safety review defaults to "caution" on uncertainty | `safety.py` |
| Strictest safety layer always wins | `safety.finalize_safety_tag()` |
| Investigation loop hard-capped at 4 iterations | `orchestrator.py` constant |
| Fix-retry hard-capped at 1 | `orchestrator.py` constant |
| Gap logs are local only — no telemetry | `gap_log.py` |
| LangChain cannot bypass the whitelist | `orchestrator.py` → `tools.run_tool()` |

---

## LLM providers

| Provider | Env var | Default model | Key needed? |
|---|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | `mistralai/mistral-7b-instruct:free` | ✅ Yes (free tier) |
| NVIDIA NIM | `NVIDIA_API_KEY` | `meta/llama-3.1-8b-instruct` | ✅ Yes (trial credits) |
| Ollama (local) | — | `llama3` | ❌ No — runs offline |

Switch by editing one line in `.env`:
```bash
LINUXAI_PROVIDER=nvidia      # NVIDIA NIM (current default)
LINUXAI_PROVIDER=openrouter  # OpenRouter cloud
LINUXAI_PROVIDER=ollama      # local Ollama — no key, fully private
```

Override the model per provider:
```bash
NVIDIA_MODEL=meta/llama-3.3-70b-instruct
OPENROUTER_MODEL=openai/gpt-4o-mini
OLLAMA_MODEL=mistral          # or gemma2, phi3, codellama, etc.
OLLAMA_BASE_URL=http://localhost:11434  # default — change for remote Ollama
```

### Using Ollama (local, private, no API key)

1. **Install Ollama:** https://ollama.com/download
2. **Pull a model:**
   ```bash
   ollama pull llama3       # recommended
   ollama pull mistral      # lighter alternative
   ollama pull gemma2       # Google's Gemma 2
   ```
3. **Set provider in `.env`:**
   ```bash
   LINUXAI_PROVIDER=ollama
   OLLAMA_MODEL=llama3
   ```
4. **Run normally** — no API key required:
   ```bash
   python3 cli.py "why is my disk full?"
   ```

Ollama uses the same OpenAI-compatible API format as NVIDIA and OpenRouter.
All three providers go through the exact same `call_llm()` code path in `llm_client.py`.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LINUXAI_PROVIDER` | `openrouter` | `openrouter` / `nvidia` / `ollama` |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter |
| `NVIDIA_API_KEY` | — | Required for NVIDIA NIM |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct:free` | Model override |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | Model override |
| `OLLAMA_MODEL` | `llama3` | Ollama model name (must be pulled locally) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL (local or remote) |
| `LINUXAI_DEBUG` | — | Print full investigation memory trace |
| `NO_COLOR` | — | Disable ANSI colour output |
