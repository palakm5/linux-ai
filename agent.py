"""
agent.py — Agentic reasoning loop for linuxai.

PUBLIC API
----------
    run_agent(query: str, provider: str = None) -> AgentResult

LOOP DESIGN
-----------
1.  First, ask the LLM whether this is a file-search query.
    If yes → call find_files once and return immediately (no loop, no diagnosis).

2.  Otherwise run up to MAX_ITERATIONS rounds:
      • LLM decides which tool to run next (plan step).
      • We validate the tool name against TOOL_REGISTRY (safety).
      • We run the tool and append {tool, result} to memory.
      • If the LLM sets "sufficient": true, break early.

3.  After the loop, one final LLM call produces the structured diagnosis:
      {"diagnosis": "...", "commands": [...], "verification_tool": "..."}

SAFETY INVARIANTS (enforced in code, not just prompt)
------------------------------------------------------
- Tool names are validated against TOOL_REGISTRY before execution.
- Loop is hard-capped at MAX_ITERATIONS regardless of LLM output.
- Caution commands are NEVER executed here — cli.py owns that flow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tools import TOOL_REGISTRY, run_tool
from llm_client import call_llm, LLMError
from gap_log import log_tool_gap
from safety import finalize_safety_tag

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4

# --------------------------------------------------------------------------- #
# Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class Command:
    cmd: str
    safety: str          # "safe" | "caution" — FINAL tag after 3-layer review
    explanation: str
    safety_reasoning: str = ""   # why it was flagged (from three-layer review)


@dataclass
class AgentResult:
    diagnosis: str
    safe_commands: List[Command]
    caution_commands: List[Command]
    verification_tool: Optional[str]
    memory: List[Dict[str, Any]]   # [{tool, args, result}, ...]  — for debug
    is_file_search: bool = False
    file_search_results: Optional[str] = None
    iterations_used: int = 0


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_TOOL_NAMES = list(TOOL_REGISTRY.keys())

_CLASSIFY_SYSTEM = """\
You are linuxai, a Linux diagnostic assistant.
Classify the user's query as either a FILE SEARCH or a DIAGNOSTIC question.

Respond with ONLY a JSON object in exactly this schema:
{
  "is_file_search": true or false,
  "find_pattern": "<natural-language pattern for find_files, or null>",
  "find_path":    "<absolute path to search under, or '/' if unclear>"
}

Rules:
- is_file_search = true when the user asks to FIND, LOCATE, or LIST files
  (e.g. "find all PDFs", "where are log files", "list Python scripts in /home").
- is_file_search = false for system health / performance / error queries.
- find_pattern and find_path are only relevant when is_file_search is true."""


_PLAN_SYSTEM = """\
You are linuxai, a Linux system diagnostic agent.
You investigate system issues step by step.

AVAILABLE TOOLS
---------------
Zero-argument tools (tool_args = {{}}):
  check_disk            - df -h: filesystem usage for all mounts
  check_memory          - free -h: RAM and swap usage
  check_processes       - top 10 processes by memory
  check_logs            - last 20 error log entries
  check_open_ports      - listening TCP/UDP ports (ss -tulwn)
  check_dirs            - du for /var/log and /home

Parameterized tools (supply tool_args as shown):
  check_directory_size(path)
      Check disk usage for one specific directory.
      Allowed paths: /var/log, /home, /tmp, /var/cache, /var/lib
      Example: {{"path": "/var/log"}}

  check_process_by_name(name)
      Investigate a specific process by name.
      name: letters, digits, dash, underscore only.
      Example: {{"name": "nginx"}}

  check_network(host)
      Ping a hostname or IP (4 packets).
      host: plain hostname or IPv4, no shell characters.
      Example: {{"host": "example.com"}}

  check_service_status(service_name)
      systemctl status for a Linux service.
      service_name: letters, digits, dash, underscore only.
      Example: {{"service_name": "nginx"}}

  find_files(pattern, path)
      Search for files by natural-language pattern.
      Example: {{"pattern": "*.log", "path": "/var/log"}}

Rules:
- next_tool MUST be one of the tool names listed above.
- Set sufficient=true when you have enough information to diagnose the problem.
- Do NOT repeat a tool already in memory unless memory is empty.
- After {max_iter} total tool calls you MUST set sufficient=true.
- For unknown/unsupported tools return sufficient=true instead.

Respond with ONLY a JSON object in exactly this schema:
{{
  "sufficient": true or false,
  "next_tool":  "<tool name or null if sufficient=true>",
  "tool_args":  {{}},
  "reasoning":  "<one sentence: what you learned and why you chose this tool>"
}}"""


_DIAGNOSE_SYSTEM = """\
You are linuxai, a Linux system diagnostic assistant.
Based on investigation findings, produce a clear, actionable diagnosis.

Respond with ONLY a JSON object in exactly this schema:
{{
  "diagnosis": "<plain-English explanation of the root cause and impact>",
  "commands": [
    {{
      "cmd":         "<exact shell command>",
      "safety":      "safe",
      "explanation": "<what this does and why>"
    }}
  ],
  "verification_tool": "<one tool name from the available list, or null>"
}}

Rules for commands:
- "safe"    = read-only, monitoring, or info commands (no system changes).
- "caution" = commands that delete files, kill processes, or modify the system.
- Order: safe commands first, caution commands last.
- Only recommend commands that are genuinely useful given the findings.
- verification_tool must be one of: {tool_names} (the tool to re-run AFTER a fix
  to confirm the problem is resolved), or null if not applicable."""


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _call_with_retry(system: str, user: str, provider: Optional[str], label: str) -> dict:
    """
    Call the LLM, retry once on JSON parse failure, then raise.
    `label` is used in error messages.
    """
    for attempt in (1, 2):
        try:
            return call_llm(system, user, provider=provider)
        except LLMError as exc:
            if attempt == 1:
                logger.warning("[%s] LLM JSON parse error (attempt 1), retrying: %s", label, exc)
            else:
                raise LLMError(f"[{label}] LLM failed after 2 attempts: {exc}") from exc


def _build_memory_text(memory: List[Dict]) -> str:
    """Serialise memory list to a compact JSON string for the prompt."""
    if not memory:
        return "[]"
    return json.dumps(memory, indent=2)


def _validate_tool(name: str) -> bool:
    return name in TOOL_REGISTRY


# --------------------------------------------------------------------------- #
# File-search fast path
# --------------------------------------------------------------------------- #

def _handle_file_search(
    classify_result: dict,
    memory: List[Dict],
    provider: Optional[str],
) -> AgentResult:
    """Run find_files once and wrap the result in an AgentResult."""
    pattern = classify_result.get("find_pattern") or ""
    path    = classify_result.get("find_path") or "/"

    print(f"  🔎 Searching: find_files(pattern={pattern!r}, path={path!r})")
    result = run_tool("find_files", {"pattern": pattern, "path": path})
    memory.append({"tool": "find_files", "args": {"pattern": pattern, "path": path}, "result": result})

    output = result.get("output") or result.get("error", "No results.")
    return AgentResult(
        diagnosis="",
        safe_commands=[],
        caution_commands=[],
        verification_tool=None,
        memory=memory,
        is_file_search=True,
        file_search_results=output,
        iterations_used=1,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_agent(query: str, provider: Optional[str] = None) -> AgentResult:
    """
    Investigate *query* using the agentic loop and return an AgentResult.

    Parameters
    ----------
    query    : The user's natural-language question.
    provider : LLM provider ("openrouter" / "nvidia") — None = use env default.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    memory: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Step 0: classify — file search or diagnostic?
    # ------------------------------------------------------------------ #
    print("\n🤔 Classifying query...")
    classify_user = f'User query: "{query}"'
    try:
        classify = _call_with_retry(_CLASSIFY_SYSTEM, classify_user, provider, "classify")
    except LLMError as exc:
        # Fallback: treat as diagnostic if classification fails
        logger.warning("Classification failed, defaulting to diagnostic: %s", exc)
        classify = {"is_file_search": False}

    if classify.get("is_file_search"):
        return _handle_file_search(classify, memory, provider)

    # ------------------------------------------------------------------ #
    # Step 1: agentic investigation loop (max MAX_ITERATIONS rounds)
    # ------------------------------------------------------------------ #
    plan_system = _PLAN_SYSTEM.format(
        tool_names=_TOOL_NAMES,
        max_iter=MAX_ITERATIONS,
    )
    iterations_used = 0
    cap_hit = False

    for iteration in range(1, MAX_ITERATIONS + 1):
        iterations_used = iteration
        memory_text = _build_memory_text(memory)
        plan_user = (
            f'Query: "{query}"\n\n'
            f"Investigation memory so far:\n{memory_text}\n\n"
            f"Iteration {iteration} of {MAX_ITERATIONS}. "
            f"Decide what to check next."
        )

        try:
            plan = _call_with_retry(plan_system, plan_user, provider, f"plan-{iteration}")
        except LLMError as exc:
            logger.error("Plan step %d failed: %s", iteration, exc)
            memory.append({"tool": "__plan_error__", "args": {}, "result": {"error": str(exc)}})
            break

        reasoning   = plan.get("reasoning", "")
        sufficient  = plan.get("sufficient", False)
        next_tool   = plan.get("next_tool")
        tool_args   = plan.get("tool_args") or {}

        if reasoning:
            print(f"\n  💭 [{iteration}/{MAX_ITERATIONS}] {reasoning}")

        # Break early if LLM says it has enough info
        if sufficient or not next_tool:
            print(f"  ✅ Sufficient information gathered after {iteration} step(s).")
            break

        # Safety: validate tool name against whitelist
        if not _validate_tool(next_tool):
            err = (
                f"LLM requested unknown tool {next_tool!r}. "
                f"Treating as observation error."
            )
            print(f"  ⚠️  {err}")
            # Log capability gap locally — never executes the tool
            log_tool_gap(
                query=query,
                memory=list(memory),
                missing_tool=next_tool,
                reason="tool not in whitelist",
            )
            print(f"  📝 Capability gap recorded. "
                  f"No suitable diagnostic tool is currently available for this step.")
            memory.append({
                "tool": next_tool,
                "args": tool_args,
                "result": {"error": err},
            })
            continue

        # Ensure tool_args is a plain dict
        if not isinstance(tool_args, dict):
            tool_args = {}

        print(f"\n  🔍 Running: {next_tool}({json.dumps(tool_args) if tool_args else ''})")
        result = run_tool(next_tool, tool_args)
        memory.append({"tool": next_tool, "args": tool_args, "result": result})

        # Show a brief snippet so the user sees live progress
        if "output" in result:
            snippet = result["output"][:300].replace("\n", " | ")
            print(f"  📊 {snippet}{'...' if len(result['output']) > 300 else ''}")
        elif "error" in result:
            print(f"  ❌ Tool error: {result['error'][:200]}")

    else:
        # Loop exhausted without breaking — cap was hit
        cap_hit = True
        print(
            f"\n  ⚠️  Iteration cap ({MAX_ITERATIONS}) reached — "
            "proceeding with gathered evidence."
        )

    # ------------------------------------------------------------------ #
    # Step 2: final diagnosis call
    # ------------------------------------------------------------------ #
    print("\n🧠 Generating diagnosis...")
    diagnose_system = _DIAGNOSE_SYSTEM.format(tool_names=_TOOL_NAMES)
    diagnose_user = (
        f'Original query: "{query}"\n\n'
        f"Full investigation memory:\n{_build_memory_text(memory)}\n\n"
        "Now produce the final diagnosis and recommended commands."
    )

    try:
        diagnosis_raw = _call_with_retry(diagnose_system, diagnose_user, provider, "diagnose")
    except LLMError as exc:
        # Hard fallback — return what we have
        return AgentResult(
            diagnosis=f"Diagnosis failed: {exc}",
            safe_commands=[],
            caution_commands=[],
            verification_tool=None,
            memory=memory,
            is_file_search=False,
            iterations_used=iterations_used,
        )

    # ------------------------------------------------------------------ #
    # Step 3: parse diagnosis output into AgentResult
    # ------------------------------------------------------------------ #
    diagnosis_text = diagnosis_raw.get("diagnosis", "No diagnosis available.")
    verification_tool = diagnosis_raw.get("verification_tool")

    # Validate verification_tool
    if verification_tool and not _validate_tool(verification_tool):
        logger.warning("LLM returned invalid verification_tool %r — ignoring.", verification_tool)
        verification_tool = None

    safe_commands: List[Command] = []
    caution_commands: List[Command] = []

    print("\n🛡️  Running three-layer safety review on commands...")
    for raw_cmd in diagnosis_raw.get("commands", []):
        if not isinstance(raw_cmd, dict):
            continue
        cmd_str      = str(raw_cmd.get("cmd", "")).strip()
        gen_safety   = str(raw_cmd.get("safety", "caution")).lower().strip()
        explain      = str(raw_cmd.get("explanation", "")).strip()
        if not cmd_str:
            continue

        # Three-layer safety: generator tag + pattern check + independent LLM review
        safety_result = finalize_safety_tag(cmd_str, gen_safety, provider=provider)
        final_tag     = safety_result["final_tag"]
        safety_reason = safety_result["reasoning"]

        c = Command(
            cmd=cmd_str,
            safety=final_tag,
            explanation=explain,
            safety_reasoning=safety_reason,
        )
        if final_tag == "safe":
            safe_commands.append(c)
        else:
            caution_commands.append(c)

    return AgentResult(
        diagnosis=diagnosis_text,
        safe_commands=safe_commands,
        caution_commands=caution_commands,
        verification_tool=verification_tool,
        memory=memory,
        is_file_search=False,
        iterations_used=iterations_used,
    )


# --------------------------------------------------------------------------- #
# Quick self-test  (python agent.py)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import os
    import sys

    print("=== linuxai agent.py self-test ===\n")

    # Check API key
    provider = os.environ.get("LINUXAI_PROVIDER", "openrouter")
    key_env = "NVIDIA_API_KEY" if provider == "nvidia" else "OPENROUTER_API_KEY"
    if not os.environ.get(key_env):
        print(f"⚠️  {key_env} not set. Set your API key to run live tests.")
        print("   Example:  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(0)

    # Choose test query from argv or use default
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "why is my disk usage high?"

    print(f"Query: {query!r}")
    print(f"Provider: {provider}\n")
    print("─" * 60)

    try:
        result = run_agent(query, provider=provider)
    except Exception as exc:
        print(f"\n❌ run_agent raised: {exc}")
        raise

    print("\n" + "═" * 60)
    if result.is_file_search:
        print("📁 FILE SEARCH RESULTS")
        print("─" * 60)
        print(result.file_search_results)
    else:
        print("🩺 DIAGNOSIS")
        print("─" * 60)
        print(result.diagnosis)

        if result.safe_commands:
            print("\n✅ SAFE COMMANDS")
            for c in result.safe_commands:
                print(f"  $ {c.cmd}")
                print(f"    → {c.explanation}")

        if result.caution_commands:
            print("\n⚠️  CAUTION COMMANDS (require confirmation)")
            for c in result.caution_commands:
                print(f"  $ {c.cmd}")
                print(f"    → {c.explanation}")

        if result.verification_tool:
            print(f"\n🔁 Verification tool: {result.verification_tool}")

    print(f"\n📋 Iterations used: {result.iterations_used}/{MAX_ITERATIONS}")
    print(f"📋 Memory entries:  {len(result.memory)}")
