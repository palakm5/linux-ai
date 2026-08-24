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


# =========================================================================== #
# FEATURE 4 — Parameterized tools (added in v2)
# =========================================================================== #

import re as _re

# --------------------------------------------------------------------------- #
# 4.1  check_directory_size — replaces fixed check_dirs with path validation
# --------------------------------------------------------------------------- #

ALLOWED_BASE_DIRS = [
    "/var/log",
    "/home",
    "/tmp",
    "/var/cache",
    "/var/lib",
]


def check_directory_size(path: str) -> Dict:
    """
    Report disk usage for a specific allowed directory.

    The path is validated against ALLOWED_BASE_DIRS before execution.
    If the path is outside the allowed scope, returns an error without
    running any subprocess.

    Command (when valid): du -sh <path>
    """
    if not path or not isinstance(path, str):
        return {"error": "check_directory_size: path must be a non-empty string"}

    # Resolve any symlinks / ".." to get the real path
    try:
        resolved = str(Path(path).resolve())
    except Exception:
        resolved = path

    # Validate against allowed base dirs (and their resolved symlink targets)
    # On macOS /var/log resolves to /private/var/log — both must be accepted
    resolved_bases = set(ALLOWED_BASE_DIRS)
    for base in ALLOWED_BASE_DIRS:
        try:
            resolved_bases.add(str(Path(base).resolve()))
        except Exception:
            pass
    allowed = any(
        resolved == base or resolved.startswith(base + "/")
        for base in resolved_bases
    )
    if not allowed:
        return {
            "error": (
                f"Path {path!r} is outside the allowed scope. "
                f"Allowed directories: {ALLOWED_BASE_DIRS}"
            )
        }

    return _run(["du", "-sh", resolved])


# --------------------------------------------------------------------------- #
# 4.2  check_process_by_name — investigate a specific process
# --------------------------------------------------------------------------- #

# Strict allowlist for process names: letters, digits, dash, underscore only
_PROC_NAME_RE = _re.compile(r'^[A-Za-z0-9_\-]{1,64}$')


def check_process_by_name(name: str) -> Dict:
    """
    Show ps entries for processes whose command matches *name*.

    The name is strictly validated: only alphanumeric, dash, and underscore
    are allowed. Shell metacharacters are rejected before any subprocess call.

    Filtering is done in Python (not via shell grep) to avoid injection risk.
    """
    if not name or not isinstance(name, str):
        return {"error": "check_process_by_name: name must be a non-empty string"}

    if not _PROC_NAME_RE.match(name):
        return {
            "error": (
                f"Invalid process name {name!r}. "
                "Only letters, digits, dash and underscore are allowed."
            )
        }

    # Run ps and filter in Python — no shell pipeline, no shell=True
    try:
        ps = subprocess.run(
            ["ps", "aux"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_DEFAULT_TIMEOUT,
            text=True,
        )
        if ps.returncode != 0:
            return {"error": f"ps failed (exit {ps.returncode}): {ps.stderr.strip()}"}

        lines = ps.stdout.splitlines()
        if not lines:
            return {"output": f"No output from ps (process {name!r} not found)"}

        header = lines[0]
        # Case-insensitive filter — match against the COMMAND column
        matched = [l for l in lines[1:] if name.lower() in l.lower()]

        if not matched:
            return {"output": f"No running processes found matching {name!r}"}

        return {"output": header + "\n" + "\n".join(matched[:20])}

    except subprocess.TimeoutExpired:
        return {"error": "check_process_by_name timed out"}
    except Exception as exc:
        return {"error": f"check_process_by_name failed: {exc}"}


# --------------------------------------------------------------------------- #
# 4.3a  check_network — ping a validated host
# --------------------------------------------------------------------------- #

# RFC-1123 hostname or dotted-decimal IPv4
_HOST_RE = _re.compile(
    r'^(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?'
    r'(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*'
    r'|(?:\d{1,3}\.){3}\d{1,3})$'
)


def check_network(host: str) -> Dict:
    """
    Ping *host* 4 times and return the result.

    The host is validated against a hostname/IPv4 regex.
    Shell metacharacters (;|&`$()><\\) are rejected before any subprocess call.
    """
    if not host or not isinstance(host, str):
        return {"error": "check_network: host must be a non-empty string"}

    host = host.strip()

    # Reject shell metacharacters explicitly
    for ch in (";", "|", "&", "`", "$", "(", ")", ">", "<", "\\", " "):
        if ch in host:
            return {"error": f"Invalid host: contains forbidden character {ch!r}"}

    if not _HOST_RE.match(host):
        return {
            "error": (
                f"Invalid hostname or IP {host!r}. "
                "Use a plain hostname (example.com) or IPv4 address (1.2.3.4)."
            )
        }

    # Use -c 4 on Linux, -c 4 on macOS (both support -c)
    return _run(["ping", "-c", "4", host], timeout=20)


# --------------------------------------------------------------------------- #
# 4.3b  check_service_status — systemctl status for a validated service
# --------------------------------------------------------------------------- #

# Same rules as process names
_SVC_NAME_RE = _re.compile(r'^[A-Za-z0-9_\-]{1,64}$')


def check_service_status(service_name: str) -> Dict:
    """
    Return the systemctl status of *service_name*.

    The service name is strictly validated: only alphanumeric, dash, and
    underscore. Shell metacharacters are rejected before any subprocess call.
    Falls back to a graceful error if systemctl is unavailable (e.g. Docker).
    """
    if not service_name or not isinstance(service_name, str):
        return {"error": "check_service_status: service_name must be a non-empty string"}

    if not _SVC_NAME_RE.match(service_name):
        return {
            "error": (
                f"Invalid service name {service_name!r}. "
                "Only letters, digits, dash and underscore are allowed."
            )
        }

    return _run(["systemctl", "status", service_name, "--no-pager"])


# --------------------------------------------------------------------------- #
# 4.3c  check_open_ports — ss -tulwn (no user parameter)
# --------------------------------------------------------------------------- #

def check_open_ports() -> Dict:
    """
    List all listening TCP and UDP ports using ss.
    No user-supplied parameter — zero injection risk.
    Falls back to netstat if ss is unavailable.
    """
    result = _run(["ss", "-tulwn"])
    if "error" not in result:
        return result
    # Fallback for systems without ss
    return _run(["netstat", "-tulwn"])


# --------------------------------------------------------------------------- #
# Register new tools in the whitelist
# --------------------------------------------------------------------------- #

TOOL_REGISTRY.update({
    "check_directory_size":  check_directory_size,
    "check_process_by_name": check_process_by_name,
    "check_network":         check_network,
    "check_service_status":  check_service_status,
    "check_open_ports":      check_open_ports,
})
