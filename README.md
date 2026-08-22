# 🐧 linuxai

AI-powered Linux diagnostic & file-search CLI tool.  
Instead of one shallow LLM call, **linuxai investigates problems step by step** — running real system commands, reading the results, and deciding what to check next — until it's confident enough to diagnose and recommend a fix.

---

## How it works

```
Your query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Classify: file search OR diagnostic?               │
└──────────────┬──────────────────────────┬───────────┘
               │ diagnostic               │ file search
               ▼                          ▼
┌──────────────────────────┐   ┌──────────────────────┐
│  Agentic loop (≤4 iters) │   │  find_files() once   │
│  LLM → pick tool         │   │  return results      │
│  run tool → read result  │   └──────────────────────┘
│  LLM → pick next tool    │
│  ... until sufficient    │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Final diagnosis call    │
│  → plain-English cause   │
│  → safe commands         │
│  → caution commands      │
│    (require [y/N])       │
│  → verification tool     │
│    (before/after diff)   │
└──────────────────────────┘
```

---

## File structure

```
linuxai/
├── cli.py            # entrypoint — argparse, display, caution-command gate
├── agent.py          # agentic loop: classify → plan → diagnose
├── tools.py          # whitelisted subprocess wrappers (6 tools)
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
pip3 install openai
```

### 2. Set your API key

**OpenRouter** (free tier available — recommended for demos):
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
python3 cli.py "find all PDF files in /home"
python3 cli.py --provider nvidia "what process is eating CPU?"
```

---

## Docker demo (full Linux environment)

### Install Docker Desktop (macOS)

1. Download from **https://www.docker.com/products/docker-desktop/**
2. Open the `.dmg`, drag Docker to Applications, launch it
3. Wait for the whale icon in the menu bar to stop animating (~30 s)
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

# Seed memory pressure (stress-ng, 70% RAM, background)
seed_problems memory

# Seed both at once
seed_problems

# Now run linuxai — it will investigate step by step
linuxai "why is my disk full?"
linuxai "my system feels slow"
linuxai "find all log files in /var"

# After you approve and run the fix, verify it worked:
# The CLI will automatically show a before/after comparison.

# Clean up seeded files
seed_problems clean
```

### Switching providers inside the container

```bash
# Use NVIDIA NIM instead
docker run -it \
  -e NVIDIA_API_KEY=$NVIDIA_API_KEY \
  -e LINUXAI_PROVIDER=nvidia \
  linuxai
```

---

## Supported tools (whitelisted — LLM cannot run arbitrary commands)

| Tool | Command | Purpose |
|---|---|---|
| `check_disk` | `df -h` | Filesystem usage |
| `check_dirs` | `du -sh /var/log /home` | Key directory sizes |
| `check_memory` | `free -h` | RAM + swap |
| `check_processes` | `ps aux --sort=-%mem` | Top 10 by memory |
| `check_logs` | `journalctl -p err` / `dmesg` | Recent errors |
| `find_files` | `find <path> -iname <glob>` | File search |

---

## Supported LLM providers

| Provider | Env var | Default model | Free tier? |
|---|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | `mistralai/mistral-7b-instruct:free` | ✅ Yes |
| NVIDIA NIM | `NVIDIA_API_KEY` | `meta/llama-3.1-8b-instruct` | ✅ Trial credits |

Switch providers without changing any code:
```bash
export LINUXAI_PROVIDER=nvidia   # or openrouter (default)
```

Override the model:
```bash
export OPENROUTER_MODEL=openai/gpt-4o-mini
export NVIDIA_MODEL=meta/llama-3.3-70b-instruct
```

---

## Safety design

| Rule | Where enforced |
|---|---|
| Caution commands need `[y/N]` confirmation | `cli.py` control flow |
| Tool names validated against whitelist before execution | `agent.py` + `tools.py` |
| Hallucinated tool names → error observation (no crash) | `agent.py` |
| Loop hard-capped at 4 iterations | `agent.py` constant |
| `shell=True` only for user-approved caution commands | `cli.py` |
| All diagnostic tools use argument lists, no shell=True | `tools.py` |

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LINUXAI_PROVIDER` | `openrouter` | LLM provider |
| `OPENROUTER_API_KEY` | — | Required for OpenRouter |
| `NVIDIA_API_KEY` | — | Required for NVIDIA NIM |
| `OPENROUTER_MODEL` | `mistralai/mistral-7b-instruct:free` | Model override |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | Model override |
| `LINUXAI_DEBUG` | — | Print full investigation memory |
| `NO_COLOR` | — | Disable ANSI colours |
