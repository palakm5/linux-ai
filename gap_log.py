"""
gap_log.py — Local gap logging for linuxai.

Records cases where the agent's planner requests a tool that does not
exist in the whitelist. This is strictly local — nothing is uploaded.

The log is append-only JSONL at tool_gap_log.jsonl in the working directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

GAP_LOG_PATH = Path("tool_gap_log.jsonl")


def log_tool_gap(
    query: str,
    memory: List[Dict[str, Any]],
    missing_tool: str = "",
    reason: str = "no suitable tool",
) -> None:
    """
    Append one entry to the local gap log.

    Parameters
    ----------
    query        : The original user query that triggered this gap.
    memory       : The agent's investigation memory at time of gap.
    missing_tool : The tool name the LLM tried to call (if known).
    reason       : Human-readable reason string.
    """
    entry: Dict[str, Any] = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "query":          query,
        "reason":         reason,
        "missing_tool":   missing_tool,
        "partial_memory": memory,   # snapshot — may be empty early in loop
    }
    try:
        with open(GAP_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        # Never let logging failure crash the agent
        import logging
        logging.getLogger(__name__).warning("gap_log write failed: %s", exc)


def read_gap_log(n: int = 20) -> List[Dict[str, Any]]:
    """
    Return the last *n* entries from the gap log (newest last).
    Returns [] if the log does not exist yet.
    """
    if not GAP_LOG_PATH.exists():
        return []
    lines = GAP_LOG_PATH.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries[-n:]


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import tempfile, os

    # Redirect log to a temp file for the test
    original = GAP_LOG_PATH
    import gap_log as _m
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    _m.GAP_LOG_PATH = tmp

    log_tool_gap(
        query="check postgres memory usage",
        memory=[{"tool": "check_disk", "result": {"output": "90% full"}}],
        missing_tool="check_postgres_memory",
        reason="tool not in whitelist",
    )

    entries = _m.read_gap_log()
    assert len(entries) == 1
    e = entries[0]
    assert e["query"] == "check postgres memory usage"
    assert e["missing_tool"] == "check_postgres_memory"
    assert "timestamp" in e
    assert "partial_memory" in e
    print("✅ gap_log.py self-test passed")
    print(f"   Entry: {json.dumps(e, indent=2)}")

    tmp.unlink(missing_ok=True)
    _m.GAP_LOG_PATH = original
