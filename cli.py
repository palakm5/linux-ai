"""
cli.py — Entrypoint for linuxai.

Usage:
    python cli.py "why is my disk full?"
    python cli.py "find all PDF files in /home"
    python cli.py --provider nvidia "memory usage is spiking"

Environment variables:
    OPENROUTER_API_KEY   — required for OpenRouter (default provider)
    NVIDIA_API_KEY       — required for NVIDIA NIM
    LINUXAI_PROVIDER     — "openrouter" (default) or "nvidia"
    OPENROUTER_MODEL     — override model for OpenRouter
    NVIDIA_MODEL         — override model for NVIDIA NIM

SAFETY INVARIANT (enforced here, not delegated to the LLM):
    Caution commands NEVER run without explicit [y/N] confirmation from the user.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
from typing import List, Optional

from agent import run_agent, AgentResult, Command
from tools import run_tool
from llm_client import get_active_provider_info, LLMError
from orchestrator import (
    run_workflow, record_fix_failure, step_verify,
    MAX_FIX_RETRIES, WorkflowState,
    step_investigate, step_diagnose, step_safety_review,
)
import orchestrator as _orch


# --------------------------------------------------------------------------- #
# Colour helpers (degrade gracefully when NO_COLOR or non-tty)
# --------------------------------------------------------------------------- #

_USE_COLOUR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def bold(t):    return _c(t, "1")
def green(t):   return _c(t, "32")
def yellow(t):  return _c(t, "33")
def red(t):     return _c(t, "31")
def cyan(t):    return _c(t, "36")
def dim(t):     return _c(t, "2")


# --------------------------------------------------------------------------- #
# Printing helpers
# --------------------------------------------------------------------------- #

_WIDTH = 68

def _rule(char: str = "─") -> str:
    return dim(char * _WIDTH)

def _header(title: str) -> None:
    print()
    print(_rule("═"))
    print(bold(f"  {title}"))
    print(_rule("═"))

def _section(title: str) -> None:
    print()
    print(_rule())
    print(bold(f"  {title}"))
    print(_rule())

def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_WIDTH, initial_indent=prefix, subsequent_indent=prefix)

def _print_command(cmd: Command, index: int) -> None:
    tag = green("✅ safe") if cmd.safety == "safe" else yellow("⚠️  caution")
    print(f"\n  [{index}] {tag}")
    print(f"      {bold('$')} {cyan(cmd.cmd)}")
    if cmd.explanation:
        print(_wrap(cmd.explanation, indent=6))
    if cmd.safety != "safe" and getattr(cmd, "safety_reasoning", ""):
        print(_wrap(f"⚠ Why flagged: {cmd.safety_reasoning}", indent=6))


# --------------------------------------------------------------------------- #
# Before/after comparison helper
# --------------------------------------------------------------------------- #

def _extract_number(text: str) -> Optional[float]:
    """
    Pull the first percentage or numeric value from a tool output string.
    Used to compare before vs after a fix (e.g. disk usage %).
    """
    # Try percentage first
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1))
    # Fall back to first plain integer
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1))
    return None


def _run_verification(tool_name: str) -> str:
    """Re-run the verification tool and return its output string."""
    result = run_tool(tool_name)
    return result.get("output") or result.get("error", "(no output)")


def _show_verification(tool_name: str, before_output: str) -> None:
    """Run the verification tool, then print a before/after comparison."""
    print(f"\n  {cyan('🔁 Re-running verification tool:')} {bold(tool_name)}")
    after_output = _run_verification(tool_name)

    before_val = _extract_number(before_output)
    after_val  = _extract_number(after_output)

    if before_val is not None and after_val is not None:
        delta = after_val - before_val
        arrow  = "↓" if delta < 0 else "↑"
        colour = green if delta < 0 else red
        print(f"\n  Before: {before_val:.1f}  →  After: {colour(f'{after_val:.1f}')}  {colour(f'({arrow}{abs(delta):.1f})')}")
    else:
        print(f"\n  {dim('Before:')}")
        for line in before_output.splitlines()[:5]:
            print(f"    {dim(line)}")
        print(f"\n  {bold('After:')}")
        for line in after_output.splitlines()[:5]:
            print(f"    {line}")


# --------------------------------------------------------------------------- #
# Caution-command confirmation + execution flow
# --------------------------------------------------------------------------- #

def _confirm_and_run(
    cmd: Command,
    index: int,
    verification_tool: Optional[str],
    before_snapshot: Optional[str],
) -> bool:
    """
    Print the caution command, ask for confirmation, run if approved.
    SAFETY: This is the only place caution commands can be executed.
    Confirmation is enforced by Python control flow, not by the LLM.

    Returns True if the command ran successfully (used for retry logic).
    """
    _print_command(cmd, index)
    print()
    try:
        answer = input("    Run this? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer != "y":
        print(f"    {dim('Skipped.')}")
        return False

    print(f"    {dim('Running...')}")
    ran_ok = False
    try:
        proc = subprocess.run(
            cmd.cmd,
            shell=True,          # caution cmds may use pipes / shell syntax
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
        )
        output = proc.stdout.strip()
        if output:
            for line in output.splitlines():
                print(f"    {line}")
        if proc.returncode != 0:
            print(red(f"    ⚠ Command exited with code {proc.returncode}"))
        else:
            ran_ok = True
    except subprocess.TimeoutExpired:
        print(red("    ✖ Command timed out after 60s"))
    except Exception as exc:
        print(red(f"    ✖ Error: {exc}"))
        return False

    # Verification: compare before/after if we have a tool and a snapshot
    if verification_tool and before_snapshot is not None:
        _show_verification(verification_tool, before_snapshot)

    return ran_ok


# --------------------------------------------------------------------------- #
# Grab a "before" snapshot for the verification tool
# --------------------------------------------------------------------------- #

def _snapshot(tool_name: Optional[str]) -> Optional[str]:
    if not tool_name:
        return None
    result = run_tool(tool_name)
    return result.get("output") or result.get("error")


# --------------------------------------------------------------------------- #
# Main display: file-search result
# --------------------------------------------------------------------------- #

def _display_file_search(result: AgentResult) -> None:
    _header("📁  File Search Results")
    if result.file_search_results:
        for line in result.file_search_results.splitlines():
            print(f"  {line}")
    else:
        print(dim("  No results found."))


# --------------------------------------------------------------------------- #
# Main display: diagnostic result
# --------------------------------------------------------------------------- #

def _display_diagnostic(result: AgentResult) -> None:
    # --- Diagnosis ---
    _header("🩺  Diagnosis")
    print(_wrap(result.diagnosis, indent=2))

    # --- Safe commands (informational) ---
    if result.safe_commands:
        _section("✅  Safe Commands  (informational — run anytime)")
        for i, cmd in enumerate(result.safe_commands, 1):
            _print_command(cmd, i)

    # --- Caution commands (require confirmation) ---
    if result.caution_commands:
        _section("⚠️   Caution Commands  (require your approval)")

        # Take a before-snapshot of the verification metric
        before = _snapshot(result.verification_tool)

        for i, cmd in enumerate(result.caution_commands, 1):
            _confirm_and_run(
                cmd,
                i,
                verification_tool=result.verification_tool,
                before_snapshot=before,
            )
    else:
        if result.verification_tool:
            _section("🔁  Verification")
            before = _snapshot(result.verification_tool)
            if before:
                _show_verification(result.verification_tool, before)

    # --- Debug trace ---
    if os.environ.get("LINUXAI_DEBUG"):
        _section("🐛  Debug: Investigation Memory")
        import json
        for entry in result.memory:
            tool  = entry.get("tool", "?")
            res   = entry.get("result", {})
            snip  = str(res)[:200]
            print(f"\n  [{tool}]  {dim(snip)}")


# --------------------------------------------------------------------------- #
# argparse setup
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="linuxai",
        description=(
            "AI-powered Linux diagnostic & file-search tool.\n"
            "Investigates system issues step by step using an agentic loop."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              linuxai "why is my disk full?"
              linuxai "memory usage keeps spiking"
              linuxai "find all log files in /var"
              linuxai --provider nvidia "what process is eating CPU?"

            Environment variables:
              OPENROUTER_API_KEY   API key for OpenRouter (default provider)
              NVIDIA_API_KEY       API key for NVIDIA NIM
              LINUXAI_PROVIDER     openrouter | nvidia  (default: openrouter)
              LINUXAI_DEBUG        Set to any value to print investigation memory
        """),
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Your question or request in natural language.",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "nvidia"],
        default=None,
        help="LLM provider to use (overrides LINUXAI_PROVIDER env var).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show current provider/model configuration and exit.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # --info flag
    if args.info:
        import json
        info = get_active_provider_info()
        print(json.dumps(info, indent=2))
        sys.exit(0)

    # Validate and join query tokens
    query = " ".join(args.query).strip()
    if not query:
        parser.error("Query must not be empty.")

    provider = args.provider  # None → agent reads from env

    # Print banner
    print()
    print(bold("  🐧 linuxai") + dim(" — AI-powered Linux diagnostic"))
    print(dim(f"  Query: {query!r}"))
    info = get_active_provider_info()
    if "error" not in info:
        print(dim(f"  Provider: {info['provider']}  |  Model: {info['model']}"))
    print()

    # Check API key early and give a clear error
    # Ollama is local and needs no key — skip the check for it
    if "error" in info or not info.get("api_key_configured"):
        prov = info.get("provider", "openrouter")
        key_env = {
            "nvidia":      "NVIDIA_API_KEY",
            "openrouter":  "OPENROUTER_API_KEY",
        }.get(prov, "OPENROUTER_API_KEY")
        print(red(f"  ✖  No API key configured for provider '{prov}'."))
        print(red(f"     Set {key_env} in your .env file or environment and retry."))
        print()
        print(dim(f"     echo '{key_env}=your-key-here' >> .env"))
        sys.exit(1)

    # Run via LangChain orchestrator
    try:
        wf_state = run_workflow(query, provider=provider)
    except LLMError as exc:
        print(red(f"\n  ✖  LLM error: {exc}"))
        sys.exit(1)
    except KeyboardInterrupt:
        print(yellow("\n\n  Interrupted."))
        sys.exit(130)
    except Exception as exc:
        print(red(f"\n  ✖  Unexpected error: {exc}"))
        if os.environ.get("LINUXAI_DEBUG"):
            raise
        sys.exit(1)

    # ── File search: display and exit ────────────────────────────────────
    if wf_state["is_file_search"]:
        _header("📁  File Search Results")
        results_text = wf_state.get("file_search_results") or "No results found."
        for line in results_text.splitlines():
            print(f"  {line}")
        print()
        print(_rule())
        print(dim(f"  Memory entries: {len(wf_state['memory'])}"))
        print(_rule())
        print()
        return

    # ── Diagnostic display ───────────────────────────────────────────────
    _header("🩺  Diagnosis")
    print(_wrap(wf_state.get("diagnosis", "No diagnosis available."), indent=2))

    # Build Command objects from workflow state (now includes safety_reasoning)
    all_cmds = wf_state.get("commands", [])
    safe_cmds    = [Command(c["cmd"], c["safety"], c.get("explanation",""),
                            c.get("safety_reasoning",""))
                    for c in all_cmds if c.get("safety") == "safe"]
    caution_cmds = [Command(c["cmd"], c["safety"], c.get("explanation",""),
                            c.get("safety_reasoning",""))
                    for c in all_cmds if c.get("safety") != "safe"]

    if safe_cmds:
        _section("✅  Safe Commands  (informational — run anytime)")
        for i, cmd in enumerate(safe_cmds, 1):
            _print_command(cmd, i)

    # ── Caution commands + bounded retry loop ────────────────────────────
    if caution_cmds:
        _section("⚠️   Caution Commands  (require your approval)")
        vtool  = wf_state.get("verification_tool")
        before = _snapshot(vtool)

        for i, cmd in enumerate(caution_cmds, 1):
            ran = _confirm_and_run(cmd, i, vtool, before)
            if not ran:
                continue

            # Verification after the fix
            if vtool and before:
                after_output = _run_verification(vtool)
                wf_state = step_verify(wf_state, after_output, before)

                if wf_state.get("resolved"):
                    print()
                    print(green("  ✅ Issue resolved!"))
                else:
                    # Fix did not work — bounded retry
                    wf_state = record_fix_failure(wf_state, cmd.cmd)
                    retry_count = wf_state["fix_retry_count"]

                    if retry_count >= MAX_FIX_RETRIES:
                        print()
                        print(red("  ✖  The issue remains unresolved after the maximum fix retry."))
                        print(red("     Manual investigation may be required."))
                    else:
                        print()
                        print(yellow(f"  ⟳  Fix did not resolve the issue. "
                                     f"Re-planning (retry {retry_count}/{MAX_FIX_RETRIES})..."))
                        # Re-enter the LangChain workflow with updated memory
                        try:
                            from orchestrator import step_investigate, step_diagnose, step_safety_review
                            wf_state = step_investigate(wf_state)
                            wf_state = step_diagnose(wf_state)
                            wf_state = step_safety_review(wf_state)

                            # Rebuild command lists from new state
                            new_cmds = wf_state.get("commands", [])
                            new_caution = [Command(c["cmd"], c["safety"],
                                                   c.get("explanation",""),
                                                   c.get("safety_reasoning",""))
                                           for c in new_cmds if c.get("safety") != "safe"]
                            if new_caution:
                                _section("⚠️   Retry: New Recommended Commands")
                                before = _snapshot(wf_state.get("verification_tool"))
                                for j, new_cmd in enumerate(new_caution, 1):
                                    _confirm_and_run(new_cmd, j,
                                                     wf_state.get("verification_tool"), before)
                            else:
                                print(dim("  No new commands recommended by retry plan."))
                        except Exception as retry_exc:
                            print(red(f"  ✖  Retry planning failed: {retry_exc}"))
    else:
        if wf_state.get("verification_tool"):
            _section("🔁  Verification")
            before = _snapshot(wf_state["verification_tool"])
            if before:
                _show_verification(wf_state["verification_tool"], before)

    # ── Debug trace ──────────────────────────────────────────────────────
    if os.environ.get("LINUXAI_DEBUG"):
        _section("🐛  Debug: Investigation Memory")
        import json
        for entry in wf_state.get("memory", []):
            tool  = entry.get("tool", "?")
            res   = entry.get("result", {})
            snip  = str(res)[:200]
            print(f"\n  [{tool}]  {dim(snip)}")

    # Footer
    print()
    print(_rule())
    iters = wf_state.get("iterations_used", 0)
    mem   = wf_state.get("memory", [])
    retries = wf_state.get("fix_retry_count", 0)
    print(dim(f"  Iterations: {iters}/{4}  |  Memory: {len(mem)}  |  Fix retries: {retries}/{MAX_FIX_RETRIES}"))
    print(_rule())
    print()


if __name__ == "__main__":
    main()
