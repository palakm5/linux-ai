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
) -> None:
    """
    Print the caution command, ask for confirmation, run if approved.
    SAFETY: This is the only place caution commands can be executed.
    Confirmation is enforced by Python control flow, not by the LLM.
    """
    _print_command(cmd, index)
    print()
    try:
        answer = input("    Run this? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer != "y":
        print(f"    {dim('Skipped.')}")
        return

    print(f"    {dim('Running...')}")
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
    except subprocess.TimeoutExpired:
        print(red("    ✖ Command timed out after 60s"))
    except Exception as exc:
        print(red(f"    ✖ Error: {exc}"))
        return

    # Verification: compare before/after if we have a tool and a snapshot
    if verification_tool and before_snapshot is not None:
        _show_verification(verification_tool, before_snapshot)


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
    if "error" in info or not info.get("api_key_configured"):
        prov = info.get("provider", "openrouter")
        key_env = "NVIDIA_API_KEY" if prov == "nvidia" else "OPENROUTER_API_KEY"
        print(red(f"  ✖  No API key configured."))
        print(red(f"     Set {key_env} in your environment and retry."))
        print()
        print(dim(f"     export {key_env}=your-key-here"))
        sys.exit(1)

    # Run the agent
    try:
        result = run_agent(query, provider=provider)
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

    # Display results
    if result.is_file_search:
        _display_file_search(result)
    else:
        _display_diagnostic(result)

    # Footer
    print()
    print(_rule())
    iters = result.iterations_used
    print(dim(f"  Iterations: {iters}/{4}  |  Memory entries: {len(result.memory)}"))
    print(_rule())
    print()


if __name__ == "__main__":
    main()
