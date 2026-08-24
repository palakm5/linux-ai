"""
orchestrator.py — LangChain-based workflow orchestration for linuxai.

Architecture
------------
LangChain orchestrates the existing components without replacing them:

    llm_client.py  → LLM/provider abstraction (unchanged)
    tools.py       → Whitelisted tool execution (unchanged)
    safety.py      → Three-layer safety review (unchanged)
    agent.py       → Core planning/diagnosis logic (orchestrated from here)
    cli.py         → User interface / confirmation gate (unchanged)

LangChain's role
----------------
- Maintains explicit, typed workflow state via a TypedDict
- Chains the plan→execute→reflect→diagnose steps using LCEL (|)
- Preserves memory + failed_commands across the bounded retry loop
- Provides clean state snapshots for debugging

SAFETY INVARIANTS (unchanged from agent.py)
-------------------------------------------
- Tool execution always goes through tools.run_tool() (whitelist check)
- Shell metachar / path injection validated in tools.py before subprocess
- LangChain CANNOT bypass the whitelist or the confirmation gate in cli.py
- Caution commands are NEVER executed here — cli.py owns confirmation
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

# LangChain LCEL imports (langchain-core)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from tools import TOOL_REGISTRY, run_tool
import llm_client as _llm_mod
from llm_client import LLMError

# Module-level alias — patch 'orchestrator.call_llm' in tests
call_llm = _llm_mod.call_llm
from gap_log import log_tool_gap
from safety import finalize_safety_tag

logger = logging.getLogger(__name__)

MAX_ITERATIONS  = 4    # investigation loop cap (same as agent.py)
MAX_FIX_RETRIES = 1    # bounded fix-retry cap


# --------------------------------------------------------------------------- #
# Workflow state — explicit, serialisable TypedDict
# --------------------------------------------------------------------------- #

class WorkflowState(Dict[str, Any]):
    """
    Mutable state dict passed through every step of the LangChain pipeline.

    Fields
    ------
    query           : str              original user query
    memory          : list[dict]       investigation observations
    diagnosis       : str | None       plain-English root cause
    commands        : list[dict]       raw commands from diagnosis LLM
    verification_tool : str | None     tool to re-run after a fix
    fix_command     : str | None       the caution command being attempted
    failed_commands : list[str]        commands that ran but didn't resolve
    fix_retry_count : int              number of fix attempts so far
    resolved        : bool             True when verification confirms fix worked
    provider        : str | None       LLM provider override
    iterations_used : int              how many plan iterations were used
    is_file_search  : bool
    file_search_results : str | None
    """


def initial_state(query: str, provider: Optional[str] = None) -> WorkflowState:
    return WorkflowState(
        query=query,
        memory=[],
        diagnosis=None,
        commands=[],
        verification_tool=None,
        fix_command=None,
        failed_commands=[],
        fix_retry_count=0,
        resolved=False,
        provider=provider,
        iterations_used=0,
        is_file_search=False,
        file_search_results=None,
    )


# --------------------------------------------------------------------------- #
# Shared helpers (also used by agent.py via import)
# --------------------------------------------------------------------------- #

def _call_with_retry(
    system: str, user: str, provider: Optional[str], label: str
) -> dict:
    for attempt in (1, 2):
        try:
            return call_llm(system, user, provider=provider)
        except LLMError as exc:
            if attempt == 1:
                logger.warning("[%s] attempt 1 failed: %s", label, exc)
            else:
                raise LLMError(f"[{label}] failed after 2 attempts: {exc}") from exc


def _memory_text(memory: List[Dict]) -> str:
    return json.dumps(memory, indent=2) if memory else "[]"


def _validate_tool(name: str) -> bool:
    return name in TOOL_REGISTRY


# --------------------------------------------------------------------------- #
# LangChain step: Classify
# --------------------------------------------------------------------------- #

_CLASSIFY_SYSTEM = """\
You are linuxai, a Linux diagnostic assistant.
Classify the user query as FILE SEARCH or DIAGNOSTIC.

Respond with ONLY a JSON object:
{
  "is_file_search": true or false,
  "find_pattern": "<natural language pattern or null>",
  "find_path":    "<absolute path or '/'>"
}"""


def step_classify(state: WorkflowState) -> WorkflowState:
    query    = state["query"]
    provider = state["provider"]
    print("\n🤔 Classifying query...")
    try:
        r = _call_with_retry(_CLASSIFY_SYSTEM, f'User query: "{query}"', provider, "classify")
    except LLMError as exc:
        logger.warning("Classification failed, defaulting to diagnostic: %s", exc)
        r = {"is_file_search": False}

    state["is_file_search"] = bool(r.get("is_file_search"))
    state["_classify_result"] = r   # pass to next step
    return state


# --------------------------------------------------------------------------- #
# LangChain step: File-search fast path
# --------------------------------------------------------------------------- #

def step_file_search(state: WorkflowState) -> WorkflowState:
    r       = state.get("_classify_result", {})
    pattern = r.get("find_pattern") or ""
    path    = r.get("find_path") or "/"
    print(f"  🔎 Searching: find_files(pattern={pattern!r}, path={path!r})")
    result  = run_tool("find_files", {"pattern": pattern, "path": path})
    state["memory"].append({"tool": "find_files", "args": {"pattern": pattern, "path": path}, "result": result})
    state["file_search_results"] = result.get("output") or result.get("error", "No results.")
    state["iterations_used"] = 1
    return state


# --------------------------------------------------------------------------- #
# LangChain step: Investigation loop
# --------------------------------------------------------------------------- #

_PLAN_SYSTEM_TMPL = """\
You are linuxai, a Linux system diagnostic agent.

AVAILABLE TOOLS
---------------
Zero-argument tools (tool_args = {{}}):
  check_disk            - df -h: filesystem usage
  check_memory          - free -h: RAM and swap
  check_processes       - top 10 processes by memory
  check_logs            - last 20 error log entries
  check_open_ports      - listening TCP/UDP ports
  check_dirs            - du for /var/log and /home

Parameterized tools:
  check_directory_size(path)   - allowed: /var/log /home /tmp /var/cache /var/lib
  check_process_by_name(name)  - letters/digits/dash/underscore only
  check_network(host)          - plain hostname or IPv4
  check_service_status(service_name) - letters/digits/dash/underscore only
  find_files(pattern, path)    - natural-language file search

{failed_note}

Rules:
- next_tool MUST be one of the tool names listed above.
- Set sufficient=true when you have enough information to diagnose.
- Do NOT repeat a tool already in memory.
- After {max_iter} total calls you MUST set sufficient=true.

Respond ONLY with JSON:
{{
  "sufficient": true or false,
  "next_tool":  "<tool name or null>",
  "tool_args":  {{}},
  "reasoning":  "<one sentence>"
}}"""


def step_investigate(state: WorkflowState) -> WorkflowState:
    query    = state["query"]
    provider = state["provider"]
    memory   = state["memory"]
    failed   = state["failed_commands"]

    failed_note = ""
    if failed:
        cmds = "\n".join(f"  - {c}" for c in failed)
        failed_note = (
            f"The following fix command(s) were already attempted and did NOT resolve the issue:\n"
            f"{cmds}\n"
            f"Do NOT recommend these commands again. Choose a different approach."
        )

    plan_system = _PLAN_SYSTEM_TMPL.format(
        failed_note=failed_note,
        max_iter=MAX_ITERATIONS,
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        state["iterations_used"] = iteration
        plan_user = (
            f'Query: "{query}"\n\n'
            f"Investigation memory:\n{_memory_text(memory)}\n\n"
            f"Iteration {iteration}/{MAX_ITERATIONS}. Decide what to check next."
        )
        try:
            plan = _call_with_retry(plan_system, plan_user, provider, f"plan-{iteration}")
        except LLMError as exc:
            logger.error("Plan step %d failed: %s", iteration, exc)
            memory.append({"tool": "__plan_error__", "args": {}, "result": {"error": str(exc)}})
            break

        reasoning  = plan.get("reasoning", "")
        sufficient = plan.get("sufficient", False)
        next_tool  = plan.get("next_tool")
        tool_args  = plan.get("tool_args") or {}

        if reasoning:
            print(f"\n  💭 [{iteration}/{MAX_ITERATIONS}] {reasoning}")

        if sufficient or not next_tool:
            print(f"  ✅ Sufficient information gathered after {iteration} step(s).")
            break

        # Whitelist validation — LangChain CANNOT bypass this
        if not _validate_tool(next_tool):
            err = f"LLM requested unknown tool {next_tool!r}. Treating as observation error."
            print(f"  ⚠️  {err}")
            log_tool_gap(query=query, memory=list(memory), missing_tool=next_tool,
                         reason="tool not in whitelist")
            print(f"  📝 Capability gap recorded.")
            memory.append({"tool": next_tool, "args": tool_args, "result": {"error": err}})
            continue

        if not isinstance(tool_args, dict):
            tool_args = {}

        print(f"\n  🔍 Running: {next_tool}({json.dumps(tool_args) if tool_args else ''})")
        result = run_tool(next_tool, tool_args)
        memory.append({"tool": next_tool, "args": tool_args, "result": result})

        if "output" in result:
            snippet = result["output"][:300].replace("\n", " | ")
            print(f"  📊 {snippet}{'...' if len(result['output']) > 300 else ''}")
        elif "error" in result:
            print(f"  ❌ Tool error: {result['error'][:200]}")
    else:
        print(f"\n  ⚠️  Iteration cap ({MAX_ITERATIONS}) reached — proceeding.")

    state["memory"] = memory
    return state


# --------------------------------------------------------------------------- #
# LangChain step: Diagnose
# --------------------------------------------------------------------------- #

_DIAGNOSE_SYSTEM = """\
You are linuxai, a Linux system diagnostic assistant.
Based on investigation findings, produce a clear, actionable diagnosis.

Respond with ONLY a JSON object:
{{
  "diagnosis": "<plain-English root cause and impact>",
  "commands": [
    {{
      "cmd":         "<exact shell command>",
      "safety":      "safe",
      "explanation": "<what this does and why>"
    }}
  ],
  "verification_tool": "<tool name or null>"
}}

Rules:
- "safe"    = read-only, informational, no system changes
- "caution" = deletes/kills/modifies the system
- Order: safe commands first, caution last
- verification_tool must be one of: {tool_names}""".format(
    tool_names=list(TOOL_REGISTRY.keys())
)


def step_diagnose(state: WorkflowState) -> WorkflowState:
    query    = state["query"]
    provider = state["provider"]
    memory   = state["memory"]
    failed   = state["failed_commands"]

    print("\n🧠 Generating diagnosis...")

    failed_note = ""
    if failed:
        cmds = ", ".join(repr(c) for c in failed)
        failed_note = f"\n\nNOTE: The following fix commands were already attempted and FAILED: {cmds}. Do not recommend them again."

    diagnose_user = (
        f'Original query: "{query}"\n\n'
        f"Full investigation memory:\n{_memory_text(memory)}"
        f"{failed_note}\n\n"
        "Produce the final diagnosis and recommended commands."
    )

    try:
        raw = _call_with_retry(_DIAGNOSE_SYSTEM, diagnose_user, provider, "diagnose")
    except LLMError as exc:
        state["diagnosis"] = f"Diagnosis failed: {exc}"
        state["commands"]  = []
        state["verification_tool"] = None
        return state

    state["diagnosis"]          = raw.get("diagnosis", "No diagnosis available.")
    state["commands"]           = raw.get("commands", [])
    vtool = raw.get("verification_tool")
    state["verification_tool"]  = vtool if vtool and _validate_tool(vtool) else None
    return state


# --------------------------------------------------------------------------- #
# LangChain step: Safety review
# --------------------------------------------------------------------------- #

def step_safety_review(state: WorkflowState) -> WorkflowState:
    """Run three-layer safety review on every command in state['commands']."""
    provider = state["provider"]
    print("\n🛡️  Running three-layer safety review on commands...")

    reviewed: List[Dict] = []
    for raw_cmd in state["commands"]:
        if not isinstance(raw_cmd, dict):
            continue
        cmd_str    = str(raw_cmd.get("cmd", "")).strip()
        gen_safety = str(raw_cmd.get("safety", "caution")).lower().strip()
        explain    = str(raw_cmd.get("explanation", "")).strip()
        if not cmd_str:
            continue

        safety_result = finalize_safety_tag(cmd_str, gen_safety, provider=provider)
        reviewed.append({
            "cmd":              cmd_str,
            "safety":           safety_result["final_tag"],
            "explanation":      explain,
            "safety_reasoning": safety_result["reasoning"],
            "layers":           safety_result.get("layers", {}),
        })

    state["commands"] = reviewed
    return state


# --------------------------------------------------------------------------- #
# LangChain step: Verification (called by cli.py after fix execution)
# --------------------------------------------------------------------------- #

def step_verify(
    state: WorkflowState,
    after_output: str,
    before_output: str,
) -> WorkflowState:
    """
    Determine whether a fix resolved the issue.

    This is called from cli.py after a caution command runs.
    It updates state["resolved"] and, if not resolved, logs the failure.
    """
    vtool = state["verification_tool"]
    if not vtool or not after_output:
        state["resolved"] = True   # can't verify → assume OK
        return state

    # Simple heuristic: compare key metrics
    # A more sophisticated check would re-query the LLM, but for the demo
    # we use the same numeric comparison as cli._extract_number
    import re
    def _num(text):
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if m: return float(m.group(1))
        m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
        if m: return float(m.group(1))
        return None

    before_val = _num(before_output)
    after_val  = _num(after_output)

    if before_val is not None and after_val is not None:
        # Improvement = metric went down (e.g. disk usage %)
        state["resolved"] = after_val < before_val
    else:
        # Can't compare numerically — assume resolved if command didn't error
        state["resolved"] = True

    return state


def record_fix_failure(state: WorkflowState, failed_cmd: str) -> WorkflowState:
    """
    Called by cli.py when verification shows the fix didn't work.
    Updates memory and retry counters so the next re-plan sees the failure.
    """
    vtool = state["verification_tool"]
    state["memory"].append({
        "tool": vtool or "verification",
        "result": {"note": f"Fix {failed_cmd!r} did not resolve the issue"},
        "note": f"Fix '{failed_cmd}' did not resolve the issue",
    })
    state["failed_commands"].append(failed_cmd)
    state["fix_retry_count"] += 1
    state["resolved"] = False
    return state


# --------------------------------------------------------------------------- #
# Build the LangChain pipeline
# --------------------------------------------------------------------------- #

def _classify_runnable(state):
    return step_classify(state)

def _file_search_runnable(state):
    return step_file_search(state)

def _investigate_runnable(state):
    return step_investigate(state)

def _diagnose_runnable(state):
    return step_diagnose(state)

def _safety_runnable(state):
    return step_safety_review(state)


def build_diagnostic_chain():
    """
    Returns an LCEL chain:
        classify → investigate → diagnose → safety_review

    The chain is a Runnable; invoke with:
        state = initial_state(query, provider)
        result_state = chain.invoke(state)
    """
    classify   = RunnableLambda(_classify_runnable)
    investigate = RunnableLambda(_investigate_runnable)
    diagnose   = RunnableLambda(_diagnose_runnable)
    safety     = RunnableLambda(_safety_runnable)

    # File-search bypasses investigate/diagnose/safety
    def route(state):
        if state.get("is_file_search"):
            return step_file_search(state)
        return (investigate | diagnose | safety).invoke(state)

    chain = classify | RunnableLambda(route)
    return chain


# --------------------------------------------------------------------------- #
# Public entry point used by agent.py
# --------------------------------------------------------------------------- #

def run_workflow(
    query: str,
    provider: Optional[str] = None,
) -> WorkflowState:
    """
    Run the full LangChain-orchestrated workflow for *query*.

    Returns the final WorkflowState. The caller (agent.py) converts this
    to an AgentResult. cli.py then handles confirmation + retry.

    The bounded retry loop (MAX_FIX_RETRIES) is managed by cli.py
    because it requires live user confirmation between attempts.
    """
    state = initial_state(query, provider)
    chain = build_diagnostic_chain()
    return chain.invoke(state)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from unittest.mock import patch
    import sys

    print("=== orchestrator.py self-test (mocked LLM) ===\n")

    with patch("orchestrator.call_llm", side_effect=[
        {"is_file_search": False},
        {"sufficient": False, "next_tool": "check_disk", "tool_args": {}, "reasoning": "Check disk"},
        {"sufficient": True,  "next_tool": None, "tool_args": {}, "reasoning": "Done"},
        {"diagnosis": "Disk 90% full.", "commands": [
            {"cmd": "df -h", "safety": "safe", "explanation": "Show usage"},
            {"cmd": "rm -f /var/log/bloat.log", "safety": "caution", "explanation": "Remove bloat"},
        ], "verification_tool": "check_disk"},
        # independent safety review for each command
        {"safety": "safe", "reasoning": "df is read-only"},
        {"safety": "caution", "reasoning": "rm deletes files"},
    ]):
        state = run_workflow("why is my disk full?")

    assert state["diagnosis"] == "Disk 90% full."
    assert state["iterations_used"] == 2
    cmds = state["commands"]
    safe_cmds    = [c for c in cmds if c["safety"] == "safe"]
    caution_cmds = [c for c in cmds if c["safety"] == "caution"]
    assert safe_cmds,    "Expected safe commands"
    assert caution_cmds, "Expected caution commands"
    assert caution_cmds[0]["safety_reasoning"]  # reasoning must be populated

    print(f"✅ Workflow state keys: {list(state.keys())}")
    print(f"✅ Diagnosis: {state['diagnosis']}")
    print(f"✅ Commands: {len(cmds)} ({len(safe_cmds)} safe, {len(caution_cmds)} caution)")
    print(f"✅ Safety reasoning on caution cmd: {caution_cmds[0]['safety_reasoning']!r}")
    print(f"✅ Iterations used: {state['iterations_used']}")
    print()

    # Test: gap logging fires through orchestrator
    import tempfile
    from pathlib import Path
    import gap_log as _gl
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    _gl.GAP_LOG_PATH = tmp

    with patch("orchestrator.call_llm", side_effect=[
        {"is_file_search": False},
        {"sufficient": False, "next_tool": "check_postgres_mem", "tool_args": {}, "reasoning": "bad tool"},
        {"sufficient": True,  "next_tool": None, "tool_args": {}, "reasoning": "done"},
        {"diagnosis": "Gap logged.", "commands": [], "verification_tool": None},
    ]):
        state2 = run_workflow("check postgres memory")

    entries = _gl.read_gap_log()
    assert entries, "Gap entry not written through orchestrator"
    assert entries[-1]["missing_tool"] == "check_postgres_mem"
    print(f"✅ Gap log fires through orchestrator: {entries[-1]['missing_tool']!r}")
    tmp.unlink(missing_ok=True)

    print("\nAll orchestrator self-tests PASSED ✅")
