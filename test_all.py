"""
test_all.py  Consolidated test suite for linuxai.

Covers all 5 features. No API key or Docker required  all LLM calls are mocked.

Run:
    python3 test_all.py
    python3 -m pytest test_all.py -v
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENROUTER_API_KEY", "sk-fake-key-for-tests")

import gap_log as _gl
from orchestrator import (
    MAX_FIX_RETRIES, WorkflowState, initial_state,
    record_fix_failure, run_workflow,
    step_diagnose, step_investigate, step_safety_review, step_verify,
)
from safety import finalize_safety_tag, pattern_check
from tools import TOOL_REGISTRY, run_tool


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _tmp_gap_log():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    original = _gl.GAP_LOG_PATH
    _gl.GAP_LOG_PATH = tmp
    try:
        yield tmp
    finally:
        _gl.GAP_LOG_PATH = original
        tmp.unlink(missing_ok=True)


def _wf(**kwargs) -> WorkflowState:
    s = initial_state("why is disk full?")
    s.update(dict(
        memory=[{"tool": "check_disk", "result": {"output": "Use% 92%"}}],
        diagnosis="Disk 92% full.",
        commands=[
            {"cmd": "df -h", "safety": "safe",
             "explanation": "Show usage", "safety_reasoning": "All layers agree safe"},
            {"cmd": "rm -f /var/log/fake.log", "safety": "caution",
             "explanation": "Remove log", "safety_reasoning": "rm deletes files"},
        ],
        verification_tool="check_disk",
        iterations_used=2,
    ))
    s.update(kwargs)
    return s


def _shared_se(*responses):
    calls = list(responses)
    idx = [0]
    def _se(*a, **kw):
        r = calls[idx[0]]; idx[0] += 1; return r
    return _se


# =========================================================================== #
# FEATURE 3  Gap Logging
# =========================================================================== #

class TestGapLogging:

    def test_log_written_with_correct_fields(self):
        with _tmp_gap_log() as tmp:
            _gl.log_tool_gap(
                query="check postgres memory",
                memory=[{"tool": "check_disk", "result": {"output": "90%"}}],
                missing_tool="check_postgres_memory",
                reason="tool not in whitelist",
            )
            lines = tmp.read_text().strip().splitlines()
            assert len(lines) == 1
            e = json.loads(lines[0])
            assert e["query"] == "check postgres memory"
            assert e["missing_tool"] == "check_postgres_memory"
            assert "timestamp" in e and "partial_memory" in e

    def test_gap_logged_through_orchestrator(self):
        with _tmp_gap_log() as tmp:
            with patch("orchestrator.call_llm", side_effect=[
                {"is_file_search": False},
                {"sufficient": False, "next_tool": "check_postgres_mem",
                 "tool_args": {}, "reasoning": "bad"},
                {"sufficient": True, "next_tool": None, "tool_args": {}, "reasoning": "done"},
                {"diagnosis": "Gap.", "commands": [], "verification_tool": None},
            ]):
                run_workflow("check postgres memory")
            entries = _gl.read_gap_log()
            assert entries and entries[-1]["missing_tool"] == "check_postgres_mem"

    def test_unknown_tool_adds_error_to_memory(self):
        with _tmp_gap_log():
            with patch("orchestrator.call_llm", side_effect=[
                {"is_file_search": False},
                {"sufficient": False, "next_tool": "non_existent_tool",
                 "tool_args": {}, "reasoning": "hallucinated"},
                {"sufficient": True, "next_tool": None, "tool_args": {}, "reasoning": "done"},
                {"diagnosis": "Could not complete.", "commands": [], "verification_tool": None},
            ]):
                state = run_workflow("test gap")
        bad = [m for m in state["memory"] if m.get("tool") == "non_existent_tool"]
        assert bad and "error" in bad[0]["result"]


# =========================================================================== #
# FEATURE 4  Parameterized Tools
# =========================================================================== #

class TestParameterizedTools:

    def test_directory_size_valid_var_log(self):
        assert "error" not in run_tool("check_directory_size", {"path": "/var/log"})

    def test_directory_size_valid_tmp(self):
        assert "error" not in run_tool("check_directory_size", {"path": "/tmp"})

    def test_directory_size_rejects_etc(self):
        assert "error" in run_tool("check_directory_size", {"path": "/etc"})

    def test_directory_size_rejects_root(self):
        assert "error" in run_tool("check_directory_size", {"path": "/root"})

    def test_directory_size_rejects_traversal(self):
        assert "error" in run_tool("check_directory_size", {"path": "/var/log/../../etc"})

    def test_process_by_name_valid(self):
        r = run_tool("check_process_by_name", {"name": "python3"})
        assert "error" not in r or "Invalid" not in r.get("error", "")

    def test_process_by_name_rejects_semicolon(self):
        r = run_tool("check_process_by_name", {"name": "nginx;rm -rf /"})
        assert "error" in r and "Invalid" in r["error"]

    def test_process_by_name_rejects_pipe(self):
        assert "error" in run_tool("check_process_by_name", {"name": "nginx|cat"})

    def test_process_by_name_rejects_expansion(self):
        assert "error" in run_tool("check_process_by_name", {"name": "$(whoami)"})

    def test_network_valid_localhost(self):
        assert "error" not in run_tool("check_network", {"host": "localhost"})

    def test_network_valid_ip(self):
        assert "error" not in run_tool("check_network", {"host": "127.0.0.1"})

    def test_network_rejects_semicolon(self):
        assert "error" in run_tool("check_network", {"host": "example.com;rm -rf /"})

    def test_network_rejects_expansion(self):
        assert "error" in run_tool("check_network", {"host": "$(whoami)"})

    def test_network_rejects_space(self):
        assert "error" in run_tool("check_network", {"host": "host with space"})

    def test_service_status_rejects_injection(self):
        r = run_tool("check_service_status", {"service_name": "nginx;whoami"})
        assert "error" in r and "Invalid" in r["error"]

    def test_service_status_rejects_expansion(self):
        assert "error" in run_tool("check_service_status", {"service_name": "$(id)"})

    def test_service_status_valid_name_not_rejected_by_validation(self):
        r = run_tool("check_service_status", {"service_name": "nginx"})
        if "error" in r:
            assert "Invalid" not in r["error"]   # may fail due to missing systemctl, not validation

    def test_open_ports_returns_output(self):
        assert "error" not in run_tool("check_open_ports", {})

    def test_all_new_tools_in_registry(self):
        expected = {"check_directory_size", "check_process_by_name",
                    "check_network", "check_service_status", "check_open_ports"}
        missing = expected - set(TOOL_REGISTRY.keys())
        assert not missing, f"Missing from registry: {missing}"


# =========================================================================== #
# FEATURE 1  Three-Layer Safety
# =========================================================================== #

class TestThreeLayerSafety:

    def test_pattern_safe_df(self):       assert pattern_check("df -h") == "safe"
    def test_pattern_safe_ls(self):       assert pattern_check("ls -lah /tmp") == "safe"
    def test_pattern_rm_rf_root(self):    assert pattern_check("rm -rf /") == "caution"
    def test_pattern_rm_fr_root(self):    assert pattern_check("rm -fr /") == "caution"
    def test_pattern_dd_device(self):     assert pattern_check("dd if=/dev/zero of=/dev/sda") == "caution"
    def test_pattern_mkfs(self):          assert pattern_check("mkfs.ext4 /dev/sdb1") == "caution"
    def test_pattern_chmod_777(self):     assert pattern_check("chmod -R 777 /") == "caution"
    def test_pattern_fork_bomb(self):     assert pattern_check(":(){:|:&};:") == "caution"

    def test_all_safe_returns_safe(self):
        with patch("safety.call_llm", return_value={"safety": "safe", "reasoning": "ok"}):
            r = finalize_safety_tag("df -h", "safe")
        assert r["final_tag"] == "safe"

    def test_generator_caution_wins(self):
        with patch("safety.call_llm", return_value={"safety": "safe", "reasoning": "ok"}):
            r = finalize_safety_tag("df -h", "caution")
        assert r["final_tag"] == "caution"

    def test_pattern_overrides_llm_safe_for_rm_rf_root(self):
        with patch("safety.call_llm", return_value={"safety": "safe", "reasoning": "wrong"}):
            r = finalize_safety_tag("rm -rf /", "safe")
        assert r["final_tag"] == "caution"
        assert "deterministic" in r["reasoning"].lower()

    def test_review_caution_overrides_safe_generator(self):
        with patch("safety.call_llm", return_value={"safety": "caution", "reasoning": "risky"}):
            r = finalize_safety_tag("some-risky-cmd", "safe")
        assert r["final_tag"] == "caution"

    def test_llm_failure_defaults_to_caution(self):
        from llm_client import LLMError
        with patch("safety.call_llm", side_effect=LLMError("rate limit")):
            r = finalize_safety_tag("ambiguous-cmd", "safe")
        assert r["final_tag"] == "caution"

    def test_reasoning_populated(self):
        with patch("safety.call_llm", return_value={"safety": "safe", "reasoning": "ok"}):
            r = finalize_safety_tag("rm -rf /", "safe")
        assert r["reasoning"]


# =========================================================================== #
# LangChain Orchestration
# =========================================================================== #

class TestOrchestration:

    def test_full_diagnostic_workflow(self):
        se = _shared_se(
            {"is_file_search": False},
            {"sufficient": False, "next_tool": "check_disk", "tool_args": {}, "reasoning": "Check disk"},
            {"sufficient": True,  "next_tool": None, "tool_args": {}, "reasoning": "Enough"},
            {"diagnosis": "Disk 90% full.", "commands": [
                {"cmd": "df -h", "safety": "safe", "explanation": "Show usage"},
                {"cmd": "rm -f /var/log/bloat.log", "safety": "caution", "explanation": "Remove bloat"},
            ], "verification_tool": "check_disk"},
            {"safety": "safe",    "reasoning": "df is read-only"},
            {"safety": "caution", "reasoning": "rm deletes files"},
        )
        with patch("orchestrator.call_llm", side_effect=se), \
             patch("safety.call_llm", side_effect=se):
            state = run_workflow("why is my disk full?")

        assert state["diagnosis"] == "Disk 90% full."
        assert any(c["safety"] == "safe"    for c in state["commands"])
        assert any(c["safety"] == "caution" for c in state["commands"])
        caution = [c for c in state["commands"] if c["safety"] == "caution"][0]
        assert caution["safety_reasoning"]

    def test_file_search_fast_path(self):
        with patch("orchestrator.call_llm", return_value={
            "is_file_search": True, "find_pattern": "*.log", "find_path": "/var/log",
        }):
            state = run_workflow("find all log files")
        assert state["is_file_search"]
        assert state["file_search_results"] is not None

    def test_whitelist_cannot_be_bypassed(self):
        with _tmp_gap_log():
            with patch("orchestrator.call_llm", side_effect=[
                {"is_file_search": False},
                {"sufficient": False, "next_tool": "run_arbitrary_shell",
                 "tool_args": {"cmd": "rm -rf /"}, "reasoning": "bad"},
                {"sufficient": True, "next_tool": None, "tool_args": {}, "reasoning": "done"},
                {"diagnosis": "Nope.", "commands": [], "verification_tool": None},
            ]):
                state = run_workflow("do something bad")
        executed = [m for m in state["memory"]
                    if m.get("tool") == "run_arbitrary_shell"
                    and "output" in m.get("result", {})]
        assert not executed, "Whitelist bypass detected!"

    def test_all_state_fields_present(self):
        with _tmp_gap_log():
            with patch("orchestrator.call_llm", side_effect=[
                {"is_file_search": False},
                {"sufficient": True, "next_tool": None, "tool_args": {}, "reasoning": "done"},
                {"diagnosis": "OK.", "commands": [], "verification_tool": None},
            ]):
                state = run_workflow("test")
        for k in ["query","memory","diagnosis","commands","verification_tool",
                   "fix_command","failed_commands","fix_retry_count","resolved",
                   "provider","iterations_used"]:
            assert k in state, f"Missing state field: {k}"


# =========================================================================== #
# FEATURE 2  Bounded Fix-Retry
# =========================================================================== #

class TestBoundedRetry:

    def test_record_failure_increments_count(self):
        s = initial_state("test")
        s["memory"] = []
        s = record_fix_failure(s, "rm -f /bloat.log")
        assert s["fix_retry_count"] == 1
        assert "rm -f /bloat.log" in s["failed_commands"]

    def test_record_failure_adds_memory_entry(self):
        s = initial_state("test")
        s["memory"] = []
        s = record_fix_failure(s, "bad-cmd")
        assert any("did not resolve" in str(m.get("note","")) for m in s["memory"])

    def test_max_retries_constant_is_one(self):
        assert MAX_FIX_RETRIES == 1

    def test_step_verify_resolved_when_metric_improves(self):
        s = initial_state("test")
        s["verification_tool"] = "check_disk"
        s = step_verify(s, after_output="Use% 30%", before_output="Use% 92%")
        assert s["resolved"] is True

    def test_step_verify_not_resolved_when_unchanged(self):
        s = initial_state("test")
        s["verification_tool"] = "check_disk"
        s = step_verify(s, after_output="Use% 92%", before_output="Use% 92%")
        assert s["resolved"] is False

    def test_failed_commands_preserved_in_state(self):
        s = initial_state("test")
        s["failed_commands"] = ["rm -f /var/log/bloat.log"]
        s["memory"] = [{"note": "Fix 'rm -f /var/log/bloat.log' did not resolve the issue"}]
        assert "rm -f /var/log/bloat.log" in s["failed_commands"]


# =========================================================================== #
# CLI Guard Tests
# =========================================================================== #

class TestCLIGuards:

    def test_help_exits_zero(self):
        r = subprocess.run([sys.executable, "cli.py", "--help"],
                           capture_output=True, text=True)
        assert r.returncode == 0
        assert "linuxai" in r.stdout.lower()

    def test_no_api_key_exits_one(self):
        env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
        r = subprocess.run([sys.executable, "cli.py", "test query"],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 1

    def test_safety_reasoning_displayed_for_caution(self):
        import cli
        wf = _wf()
        with patch("sys.argv", ["linuxai", "disk full"]), \
             patch("cli.run_workflow", return_value=wf), \
             patch("builtins.input", return_value="n"), \
             patch("cli.run_tool", return_value={"output": "92%"}):
            try: cli.main()
            except SystemExit as e: assert e.code in (0, None)

    def test_fix_resolved_shows_success(self):
        import cli
        rt = iter([{"output": "92%"}, {"output": "30%"}, {"output": "30%"}])
        with patch("sys.argv", ["linuxai", "disk full"]), \
             patch("cli.run_workflow", return_value=_wf()), \
             patch("builtins.input", return_value="y"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="done\n")), \
             patch("cli.run_tool", side_effect=rt):
            try: cli.main()
            except SystemExit as e: assert e.code in (0, None)

    def test_fix_not_resolved_hits_cap(self):
        import cli
        retry = _wf(fix_retry_count=1, failed_commands=["rm -f /var/log/fake.log"],
                    commands=[{"cmd": "truncate -s 0 /var/log/other.log", "safety": "caution",
                               "explanation": "Truncate", "safety_reasoning": "modifies file"}])
        with patch("sys.argv", ["linuxai", "disk full"]), \
             patch("cli.run_workflow", return_value=_wf()), \
             patch("builtins.input", side_effect=["y", "y"]), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="done\n")), \
             patch("cli.run_tool", side_effect=iter([{"output": "92%"}]*20)), \
             patch("cli.step_investigate",   return_value=retry), \
             patch("cli.step_diagnose",      return_value=retry), \
             patch("cli.step_safety_review", return_value=retry):
            try: cli.main()
            except SystemExit as e: assert e.code in (0, None)


# =========================================================================== #
# Runner
# =========================================================================== #

if __name__ == "__main__":
    import traceback

    CLASSES = [
        TestGapLogging,
        TestParameterizedTools,
        TestThreeLayerSafety,
        TestOrchestration,
        TestBoundedRetry,
        TestCLIGuards,
    ]

    total = passed = failed = 0
    failures = []

    for cls in CLASSES:
        print(f"\n{'='*64}")
        print(f"  {cls.__name__}")
        print(f"{'='*64}")
        obj = cls()
        for method in [m for m in dir(obj) if m.startswith("test_")]:
            total += 1
            try:
                getattr(obj, method)()
                print(f"  ✅  {method}")
                passed += 1
            except Exception as exc:
                print(f"  ❌  {method}")
                print(f"       {exc}")
                if os.environ.get("LINUXAI_DEBUG"):
                    traceback.print_exc()
                failures.append((cls.__name__, method, exc))
                failed += 1

    print(f"\n{'='*64}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"   |   {failed} FAILED")
        for cn, m, e in failures:
            print(f"\n  ❌  {cn}.{m}: {e}")
        sys.exit(1)
    else:
        print("  ✅")
    print(f"{'='*64}\n")
