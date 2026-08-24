"""
safety.py — Three-layer safety classification for linuxai.

Every recommended command passes through three independent checks.
The strictest result always wins: safe + safe + safe → safe,
but safe + caution + safe → caution.

Layer 1: generator_tag   — the diagnosis LLM's own self-assessment
Layer 2: pattern_check   — deterministic regex (no LLM, always runs)
Layer 3: independent_safety_review — a second LLM call with zero context
                                      about why the command was generated

All LLM calls go through the existing call_llm() from llm_client.py.
No new LLM client is created here.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

from llm_client import call_llm, LLMError

# --------------------------------------------------------------------------- #
# Layer 2 — Deterministic pattern check
# --------------------------------------------------------------------------- #

# Patterns that flag a command as ALWAYS caution, regardless of LLM opinion.
# Ordered from most to least specific.
DANGEROUS_PATTERNS = [
    r"rm\s+-[rRfF]*[fF][rR]?\s+/(?!\S)",   # rm -rf / or rm -fr /
    r"rm\s+-[rRfF]+\s+/\s*$",               # rm -rf / (trailing space)
    r"\bdd\b.*of=/dev/",                    # dd overwriting a device
    r"\bmkfs\b",                             # format a filesystem
    r"chmod\s+-[rR]\s+777\s+/",             # chmod -R 777 /
    r":\(\)\s*\{.*:\|:&",                    # fork bomb
    r"\bsystemctl\s+(?:disable|mask|stop)\b",  # stopping critical services
    r"\bkillall\b",                          # mass process kill
    r">\s*/dev/sd[a-z]",                    # redirect into a block device
    r"rm\s+.*-[rRfF].*\s+/(?:etc|boot|usr|lib|bin|sbin)\b",  # wipe system dirs
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


def pattern_check(command: str) -> str:
    """
    Deterministic safety check — no LLM involved.

    Returns "caution" if any dangerous pattern matches, else "safe".
    This is the backstop that can NEVER be fooled by a hallucinating LLM.
    """
    for pattern in _COMPILED:
        if pattern.search(command):
            return "caution"
    return "safe"


# --------------------------------------------------------------------------- #
# Layer 3 — Independent LLM safety review
# --------------------------------------------------------------------------- #

_REVIEW_SYSTEM = """\
You are a security-focused Linux command reviewer.
You will be shown a single shell command. Assess whether it is safe to run.

Rules:
- "safe"    = read-only, informational, or low-risk changes on non-critical paths.
- "caution" = deletes files, kills processes, modifies system config, or has
              irreversible effects.
- When in doubt, return "caution". Uncertainty must never become "safe".
- You receive ZERO context about why the command was generated. Judge it in isolation.

Respond with ONLY a JSON object in exactly this schema:
{
  "safety": "safe" or "caution",
  "reasoning": "<one sentence explaining your decision>"
}"""


def independent_safety_review(
    command: str,
    provider: Optional[str] = None,
) -> Dict:
    """
    Call the LLM via the existing call_llm() to review ONE command in isolation.

    The LLM sees only the command and the review instructions — zero context
    about the user's query or why the command was suggested.

    When LINUXAI_PROVIDER=ollama, the safety review uses OLLAMA_SAFETY_MODEL
    (falls back to OLLAMA_MODEL if not set). This lets you run a lighter local
    model (e.g. mistral) for the binary safe/caution call while the main
    planning/diagnosis uses a heavier one (e.g. llama3).

    Returns:
        {"safety": "safe"|"caution", "reasoning": "<one sentence>"}

    On LLM failure, defaults to "caution" (fail safe).
    """
    user_prompt = f'Command to review:\n\n  {command}\n\nAssess whether this command is safe or caution.'

    # Resolve the safety-specific model override
    # For Ollama: OLLAMA_SAFETY_MODEL → OLLAMA_MODEL → provider default
    # For cloud providers: no separate safety model (same model, same cost per token)
    active_provider = provider or os.environ.get("LINUXAI_PROVIDER", "openrouter")
    safety_model: Optional[str] = None
    if active_provider == "ollama":
        safety_model = (
            os.environ.get("OLLAMA_SAFETY_MODEL", "").strip()
            or os.environ.get("OLLAMA_MODEL", "").strip()
            or None          # fall back to provider default (llama3)
        )

    try:
        result = call_llm(
            system_prompt=_REVIEW_SYSTEM,
            user_prompt=user_prompt,
            provider=provider,
            model=safety_model,      # None → use provider default; set → override
            temperature=0.0,         # maximum determinism for safety review
            max_tokens=256,
        )
        safety = str(result.get("safety", "caution")).lower().strip()
        reasoning = str(result.get("reasoning", "")).strip()
        # Normalise to safe/caution only
        if safety not in ("safe", "caution"):
            safety = "caution"
        return {"safety": safety, "reasoning": reasoning}
    except LLMError:
        # Fail safe — if we can't review, default to caution
        return {
            "safety": "caution",
            "reasoning": "Safety review unavailable; defaulting to caution.",
        }


# --------------------------------------------------------------------------- #
# Final decision — strictest wins
# --------------------------------------------------------------------------- #

def finalize_safety_tag(
    command: str,
    generator_tag: str,
    provider: Optional[str] = None,
) -> Dict:
    """
    Run all three safety layers and return the strictest result.

    Parameters
    ----------
    command       : The shell command string to evaluate.
    generator_tag : The tag assigned by the diagnosis LLM ("safe"|"caution").
    provider      : LLM provider for the independent review call.

    Returns
    -------
    {
        "final_tag": "safe" | "caution",
        "reasoning": "<explanation of why it was flagged, or 'All layers agree: safe'>",
        "layers": {
            "generator": "safe"|"caution",
            "pattern":   "safe"|"caution",
            "review":    "safe"|"caution",
        }
    }
    """
    generator_tag = (generator_tag or "caution").lower().strip()
    if generator_tag not in ("safe", "caution"):
        generator_tag = "caution"

    # Layer 2: deterministic (always runs, never fails)
    pattern_result = pattern_check(command)

    # Layer 3: independent LLM review
    review = independent_safety_review(command, provider=provider)
    review_result = review["safety"]
    review_reason = review["reasoning"]

    layers = {
        "generator": generator_tag,
        "pattern":   pattern_result,
        "review":    review_result,
    }

    # Strictest wins: any "caution" makes the final tag "caution"
    if any(v == "caution" for v in layers.values()):
        # Determine the primary reason to show the user
        if pattern_result == "caution":
            reasoning = (
                "Flagged by deterministic safety rules: "
                "the command matches a known dangerous pattern."
            )
        elif review_result == "caution":
            reasoning = review_reason or "Flagged by independent safety review."
        else:
            reasoning = "Flagged as caution by the generating model."
        return {"final_tag": "caution", "reasoning": reasoning, "layers": layers}

    return {
        "final_tag": "safe",
        "reasoning": "All three safety layers agree this command is safe.",
        "layers": layers,
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json

    print("=== safety.py self-test (deterministic layer only) ===\n")

    cases = [
        # (command, generator_tag, expected_final)
        ("ls -lah /tmp",            "safe",    "safe"),     # obviously safe
        ("df -h",                   "safe",    "safe"),     # obviously safe
        ("rm -rf /",                "safe",    "caution"),  # pattern catches it
        ("rm -rf / ",               "safe",    "caution"),  # trailing space variant
        ("dd if=/dev/zero of=/dev/sda", "safe","caution"), # pattern catches it
        ("mkfs.ext4 /dev/sdb1",     "safe",    "caution"),  # pattern catches it
        ("chmod -R 777 /",          "safe",    "caution"),  # pattern catches it
        ("rm -f /var/log/fake.log", "caution", "caution"),  # generator says caution
        ("df -h",                   "caution", "caution"),  # generator says caution → caution
        (":(){:|:&};:",             "safe",    "caution"),  # fork bomb
    ]

    all_pass = True
    for cmd, gen, expected in cases:
        # Only test deterministic layer here (skip LLM review in self-test)
        pat = pattern_check(cmd)
        layers = {"generator": gen, "pattern": pat, "review": "safe"}
        final = "caution" if any(v == "caution" for v in layers.values()) else "safe"
        ok = (final == expected)
        if not ok:
            all_pass = False
        print(f"{'✅' if ok else '❌'} {cmd[:50]!r:52s} → {final} (expected {expected})")
        if not ok:
            print(f"   layers: {layers}")

    print()
    print("All deterministic tests PASSED ✅" if all_pass else "SOME TESTS FAILED ❌")
