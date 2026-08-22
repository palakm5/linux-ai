"""
tools.py — Whitelisted diagnostic command wrappers for linuxai.

SAFETY CONTRACT
---------------
- Only the functions explicitly defined here may be invoked by the agent.
- No function ever runs an arbitrary shell string; every command is fully
  constructed from hard-coded templates + validated arguments.
- All functions return a plain dict:  {"output": "..."}  OR  {"error": "..."}
  They NEVER raise — callers can always safely read the dict.

Compatible with Python 3.9+.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional, Callable

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

_DEFAULT_TIMEOUT = 30  # seconds


def _run(cmd: list, timeout: int = _DEFAULT_TIMEOUT) -> Dict:
    """
    Run *cmd* (a pre-built argument list) and return a structured result.

    Returns:
        {"output": "<stdout+stderr combined>"}  on success (rc == 0)
        {"error":  "<description>"}             on timeout / non-zero exit
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            timeout=timeout,
            text=True,
        )
        if result.returncode == 0:
            return {"output": result.stdout.strip()}
        else:
            return {
                "error": (
                    f"Command exited with code {result.returncode}.\n"
                    f"Output:\n{result.stdout.strip()}"
                )
            }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s: {shlex.join(cmd)}"}
    except FileNotFoundError:
        return {"error": f"Executable not found: {cmd[0]!r}"}
    except Exception as exc:  # pragma: no cover
        return {"error": f"Unexpected error running {shlex.join(cmd)}: {exc}"}


# --------------------------------------------------------------------------- #
# Whitelisted diagnostic tools
# --------------------------------------------------------------------------- #

def check_disk() -> Dict:
    """
    Report disk usage for all mounted filesystems.
    Command: df -h
    """
    return _run(["df", "-h"])


def check_dirs() -> Dict:
    """
    Show the disk usage of key directories (/var/log and /home).
    Command: du -sh /var/log /home 2>/dev/null | sort -rh

    Note: The pipe is implemented in Python to avoid shell=True.
    """
    try:
        du = subprocess.run(
            ["du", "-sh", "/var/log", "/home"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_DEFAULT_TIMEOUT,
            text=True,
        )
        # sort -rh (reverse human-readable sort)
        sort = subprocess.run(
            ["sort", "-rh"],
            input=du.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_DEFAULT_TIMEOUT,
            text=True,
        )
        return {"output": sort.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"error": "check_dirs timed out"}
    except Exception as exc:
        return {"error": f"check_dirs failed: {exc}"}


def check_memory() -> Dict:
    """
    Report memory and swap usage.
    Command: free -h
    """
    return _run(["free", "-h"])


def check_processes() -> Dict:
    """
    Show the top 10 processes by memory consumption.

    Linux:  ps aux --sort=-%mem  (GNU ps)
    macOS:  ps aux -m            (BSD ps, sorts descending by memory)

    The pipe is implemented in Python to avoid shell=True.
    """
    # Try Linux-style sort first; if it fails try macOS-style
    for cmd in (["ps", "aux", "--sort=-%mem"], ["ps", "aux", "-m"]):
        try:
            ps = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=_DEFAULT_TIMEOUT,
                text=True,
            )
            if ps.returncode == 0 and ps.stdout.strip():
                lines = ps.stdout.splitlines()
                output = chr(10).join(lines[:10])
                return {"output": output}
        except subprocess.TimeoutExpired:
            return {"error": "check_processes timed out"}
        except Exception:
            continue  # try next variant

    return {"error": "check_processes: ps command failed on this platform"}



def check_logs() -> Dict:
    """
    Show the last 20 error-level system log entries.
    Primary:  journalctl -p err -n 20 --no-pager
    Fallback: dmesg | tail -20  (for minimal containers without systemd)
    """
    # Try journalctl first
    try:
        jctl = subprocess.run(
            ["journalctl", "-p", "err", "-n", "20", "--no-pager"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_DEFAULT_TIMEOUT,
            text=True,
        )
        if jctl.returncode == 0 and jctl.stdout.strip():
            return {"output": jctl.stdout.strip()}
    except FileNotFoundError:
        pass  # journalctl not installed — fall through to dmesg
    except subprocess.TimeoutExpired:
        return {"error": "check_logs timed out (journalctl)"}

    # journalctl unavailable or empty — fall back to dmesg
    try:
        dmesg = subprocess.run(
            ["dmesg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_DEFAULT_TIMEOUT,
            text=True,
        )
        lines = dmesg.stdout.splitlines()
        tail = "\n".join(lines[-20:]) if lines else "(dmesg produced no output)"
        return {"output": f"[journalctl unavailable — showing dmesg tail]\n{tail}"}
    except subprocess.TimeoutExpired:
        return {"error": "check_logs timed out (both journalctl and dmesg)"}
    except FileNotFoundError:
        return {"error": "Neither journalctl nor dmesg is available on this system."}
    except Exception as exc:
        return {"error": f"check_logs failed: {exc}"}


# --------------------------------------------------------------------------- #
# File search tool
# --------------------------------------------------------------------------- #

# Map human-readable file-type keywords → glob patterns
_TYPE_MAP: Dict[str, str] = {
    "pdf":    "*.pdf",
    "pdfs":   "*.pdf",
    "image":  "*.jpg",
    "images": "*.jpg",
    "photo":  "*.jpg",
    "photos": "*.jpg",
    "video":  "*.mp4",
    "videos": "*.mp4",
    "log":    "*.log",
    "logs":   "*.log",
    "text":   "*.txt",
    "csv":    "*.csv",
    "zip":    "*.zip",
    "python": "*.py",
    "script": "*.sh",
}

# Map human-readable time keywords → -mtime N values (days)
_TIME_MAP: Dict[str, str] = {
    "today":         "-1",
    "yesterday":     "-2",
    "last 24 hours": "-1",
    "last 3 days":   "-3",
    "last week":     "-7",
    "last month":    "-30",
    "last 7 days":   "-7",
    "last 30 days":  "-30",
    "this week":     "-7",
    "this month":    "-30",
    "recent":        "-7",
}


def find_files(pattern: str = "", path: str = "/", max_results: int = 50) -> Dict:
    """
    Search for files matching *pattern* under *path*.

    *pattern* is a natural-language or glob string.  Examples:
        "*.pdf"               — used directly as a glob
        "PDFs"                — resolved via _TYPE_MAP to "*.pdf"
        "PDFs edited last week" — resolved to glob + -mtime -7

    Returns up to *max_results* matches.
    """
    # Validate path
    search_path = path if path and Path(path).exists() else "/"

    pattern_lower = pattern.lower()

    # ---- resolve glob -------------------------------------------------------
    glob_pattern: Optional[str] = None
    for keyword, glob in _TYPE_MAP.items():
        if keyword in pattern_lower:
            glob_pattern = glob
            break
    if glob_pattern is None and pattern.strip():
        glob_pattern = pattern.strip()

    # ---- resolve time filter ------------------------------------------------
    mtime_val: Optional[str] = None
    for keyword, mtime in _TIME_MAP.items():
        if keyword in pattern_lower:
            mtime_val = mtime
            break

    # ---- build find command -------------------------------------------------
    cmd: list = ["find", search_path, "-type", "f"]

    if glob_pattern:
        cmd += ["-iname", glob_pattern]

    if mtime_val:
        cmd += ["-mtime", mtime_val]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # suppress "permission denied" noise
            timeout=60,
            text=True,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return {"output": "No files found matching the given criteria."}
        truncated = lines[:max_results]
        suffix = (
            f"\n... (showing {max_results} of {len(lines)} results)"
            if len(lines) > max_results
            else ""
        )
        return {"output": "\n".join(truncated) + suffix}
    except subprocess.TimeoutExpired:
        return {"error": "find_files timed out (search may be too broad — specify a narrower path)"}
    except Exception as exc:
        return {"error": f"find_files failed: {exc}"}


# --------------------------------------------------------------------------- #
# Tool registry — the ONLY names the agent is allowed to invoke
# --------------------------------------------------------------------------- #

TOOL_REGISTRY: Dict[str, Callable] = {
    "check_disk":      check_disk,
    "check_dirs":      check_dirs,
    "check_memory":    check_memory,
    "check_processes": check_processes,
    "check_logs":      check_logs,
    "find_files":      find_files,
}


def run_tool(name: str, args: Optional[Dict] = None) -> Dict:
    """
    Safe dispatch: look up *name* in TOOL_REGISTRY and call it.

    Returns {"error": "..."} if the name is not whitelisted — never raises.
    """
    if name not in TOOL_REGISTRY:
        return {
            "error": (
                f"Tool {name!r} is not in the whitelist. "
                f"Available tools: {list(TOOL_REGISTRY.keys())}"
            )
        }
    fn = TOOL_REGISTRY[name]
    try:
        return fn(**(args or {}))
    except TypeError as exc:
        return {"error": f"Bad arguments for tool {name!r}: {exc}"}


# --------------------------------------------------------------------------- #
# Quick self-test (run with:  python tools.py)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json

    tests = [
        ("check_disk",      {}),
        ("check_dirs",      {}),
        ("check_memory",    {}),
        ("check_processes", {}),
        ("check_logs",      {}),
        ("find_files",      {"pattern": "*.py", "path": "/tmp"}),
        ("bad_tool",        {}),  # should return error, not crash
    ]

    for tool_name, tool_args in tests:
        print(f"\n{'='*60}")
        print(f"▶  {tool_name}({tool_args})")
        print("─" * 60)
        result = run_tool(tool_name, tool_args)
        # Truncate long output for readability
        if "output" in result and len(result["output"]) > 500:
            result = {"output": result["output"][:500] + "\n... [truncated]"}
        print(json.dumps(result, indent=2))
