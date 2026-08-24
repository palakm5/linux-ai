"""
llm_client.py — Thin LLM wrapper for linuxai.

Supports two provider paths, selectable at runtime via the LINUXAI_PROVIDER
environment variable (or the `provider` argument):

  "openrouter"  — OpenRouter.ai  (default)
                  Requires: OPENROUTER_API_KEY
                  Model:    OPENROUTER_MODEL  (default: mistralai/mistral-7b-instruct:free)

  "nvidia"      — NVIDIA NIM / build.nvidia.com
                  Requires: NVIDIA_API_KEY
                  Model:    NVIDIA_MODEL  (default: meta/llama-3.1-8b-instruct)

Both providers expose an OpenAI-compatible REST endpoint, so we use the
openai Python SDK with a custom base_url for each.

The single public function is:

    call_llm(system_prompt, user_prompt,
             provider=None, model=None) -> dict

It:
  1. Selects the right client + model based on the provider.
  2. Sends the request, instructing the model to return ONLY valid JSON.
  3. Strips any accidental ```json ... ``` fences from the response.
  4. Parses and returns the JSON dict.
  5. Raises LLMError (a clear, catchable exception) if parsing fails,
     so the agent loop can retry once before giving up.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Optional

# ---- Load .env automatically (project root or any parent directory) ---------
try:
    from dotenv import load_dotenv
    load_dotenv()          # looks for .env in cwd and walks up the tree
except ImportError:
    pass                   # python-dotenv not installed — fall back to plain env vars

# ---- openai SDK (both providers use the same SDK with different base_url) ---
try:
    import openai
except ImportError as exc:
    raise ImportError(
        "openai package is required: pip install openai"
    ) from exc


# --------------------------------------------------------------------------- #
# Custom exception
# --------------------------------------------------------------------------- #

class LLMError(Exception):
    """Raised when the LLM returns something that can't be parsed as JSON."""
    pass


# --------------------------------------------------------------------------- #
# Provider configuration
# --------------------------------------------------------------------------- #

_PROVIDERS = {
    "openrouter": {
        "base_url":    "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env":   "OPENROUTER_MODEL",
        "default_model": "mistralai/mistral-7b-instruct:free",
        "requires_key": True,
    },
    "nvidia": {
        "base_url":    "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "model_env":   "NVIDIA_MODEL",
        "default_model": "meta/llama-3.1-8b-instruct",
        "requires_key": True,
    },
    "ollama": {
        # base_url is overridable via OLLAMA_BASE_URL for remote/custom setups
        "base_url":    "http://localhost:11434/v1",
        "base_url_env": "OLLAMA_BASE_URL",
        "api_key_env": "",            # Ollama has no API key
        "model_env":   "OLLAMA_MODEL",
        "default_model": "llama3",    # override with OLLAMA_MODEL=mistral etc.
        "requires_key": False,
    },
}

_DEFAULT_PROVIDER = "openrouter"


def _get_provider_name(provider: Optional[str] = None) -> str:
    """Resolve provider: arg > env > default."""
    if provider:
        name = provider.lower().strip()
    else:
        name = os.environ.get("LINUXAI_PROVIDER", _DEFAULT_PROVIDER).lower().strip()
    if name not in _PROVIDERS:
        raise LLMError(
            f"Unknown provider {name!r}. "
            f"Choose from: {list(_PROVIDERS.keys())}"
        )
    return name


def _build_client(provider_name: str) -> tuple:
    """
    Build an openai.OpenAI client configured for the given provider.
    Returns (client, model_name).

    Ollama note: Ollama's REST API is OpenAI-compatible. No real API key
    is required — we pass the literal string "ollama" so the SDK is happy.
    The base URL can be overridden via OLLAMA_BASE_URL for remote setups.
    """
    cfg = _PROVIDERS[provider_name]

    # ── API key ──────────────────────────────────────────────────────────
    if cfg.get("requires_key", True):
        api_key = os.environ.get(cfg["api_key_env"], "").strip()
        if not api_key:
            raise LLMError(
                f"Missing API key for provider {provider_name!r}. "
                f"Set the {cfg['api_key_env']} environment variable."
            )
    else:
        # Ollama: no real key needed — use a placeholder the SDK accepts
        api_key = os.environ.get(cfg.get("api_key_env", ""), "ollama") or "ollama"

    # ── Base URL (Ollama allows override for remote/proxied instances) ────
    base_url = cfg["base_url"]
    if "base_url_env" in cfg:
        base_url = os.environ.get(cfg["base_url_env"], base_url).rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"

    # ── Model ─────────────────────────────────────────────────────────────
    model = os.environ.get(cfg["model_env"], cfg["default_model"]).strip()

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return client, model


# --------------------------------------------------------------------------- #
# JSON fence stripper
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


def _strip_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` Markdown fences from *text*.
    If fences are found, return the inner content; otherwise return *text* as-is.
    """
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    **kwargs,           # forwarded verbatim to client.chat.completions.create()
                        # e.g. extra_body={"options": {"num_ctx": 4096}} for Ollama
) -> dict:
    """
    Call the LLM and return a parsed JSON dict.

    Parameters
    ----------
    system_prompt : str
        The system-level instructions (should tell the model to respond with
        ONLY valid JSON, no preamble).
    user_prompt : str
        The user turn / question.
    provider : str, optional
        "openrouter" or "nvidia". Falls back to LINUXAI_PROVIDER env var,
        then to "openrouter".
    model : str, optional
        Override the model name. Falls back to provider-specific env var /
        default.
    temperature : float
        LLM temperature (default 0.2 for deterministic JSON output).
    max_tokens : int
        Max tokens to generate.

    Returns
    -------
    dict
        Parsed JSON response from the LLM.

    Raises
    ------
    LLMError
        If the response cannot be parsed as JSON, or if an API/auth error
        occurs.
    """
    provider_name = _get_provider_name(provider)
    client, resolved_model = _build_client(provider_name)
    if model:
        resolved_model = model

    # Always embed a JSON reminder in the system prompt
    json_reminder = textwrap.dedent("""\

        IMPORTANT: Your response MUST be ONLY valid JSON — no preamble,
        no explanation, no markdown fences. Start your reply with '{' and
        end with '}'. Do not include any text before or after the JSON object.
    """)
    full_system = system_prompt.rstrip() + json_reminder

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    except openai.AuthenticationError as exc:
        raise LLMError(f"Authentication failed for {provider_name!r}: {exc}") from exc
    except openai.RateLimitError as exc:
        raise LLMError(f"Rate limit hit on {provider_name!r}: {exc}") from exc
    except openai.APIConnectionError as exc:
        raise LLMError(f"Cannot reach {provider_name!r} API: {exc}") from exc
    except openai.APIStatusError as exc:
        raise LLMError(
            f"API error from {provider_name!r} (HTTP {exc.status_code}): {exc.message}"
        ) from exc
    except Exception as exc:
        raise LLMError(f"Unexpected LLM error: {exc}") from exc

    # Extract raw text
    raw = response.choices[0].message.content or ""

    # Defensively strip any markdown fences the model snuck in
    cleaned = _strip_fences(raw)

    # Parse JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"LLM response is not valid JSON.\n"
            f"Raw response:\n{raw}\n"
            f"Parse error: {exc}"
        ) from exc


def get_active_provider_info() -> dict:
    """
    Return a summary of the currently configured provider & model.
    Useful for CLI diagnostics / --info flag.
    """
    try:
        provider_name = _get_provider_name()
        cfg = _PROVIDERS[provider_name]
        model = os.environ.get(cfg["model_env"], cfg["default_model"])

        # Base URL (Ollama may be overridden)
        base_url = cfg["base_url"]
        if "base_url_env" in cfg:
            base_url = os.environ.get(cfg["base_url_env"], base_url)

        # API key status — Ollama never needs one
        if cfg.get("requires_key", True):
            api_key_set = bool(os.environ.get(cfg["api_key_env"], "").strip())
        else:
            api_key_set = True   # local Ollama needs no key — always "configured"

        return {
            "provider":          provider_name,
            "base_url":          base_url,
            "model":             model,
            "api_key_configured": api_key_set,
        }
    except LLMError as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
# Quick self-test  (python llm_client.py)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    print("=== linuxai LLM client self-test ===\n")

    info = get_active_provider_info()
    print(f"Provider info: {json.dumps(info, indent=2)}\n")

    if not info.get("api_key_configured"):
        print("⚠️  No API key found — skipping live call test.")
        print("   Set OPENROUTER_API_KEY (or NVIDIA_API_KEY + LINUXAI_PROVIDER=nvidia)")
        sys.exit(0)

    # Test 1: happy path — model returns clean JSON
    print("--- Test 1: valid JSON response ---")
    SYSTEM = (
        "You are a helpful assistant that responds ONLY with valid JSON objects."
    )
    USER = (
        'Return a JSON object with two fields: "greeting" (say hello) '
        'and "number" (the integer 42).'
    )
    try:
        result = call_llm(SYSTEM, USER)
        print(f"✅ Parsed OK: {result}")
    except LLMError as e:
        print(f"❌ LLMError: {e}")

    print()

    # Test 2: simulate a fenced response (internal unit test of _strip_fences)
    print("--- Test 2: fence stripper unit test ---")
    fenced = '```json\n{"key": "value", "num": 1}\n```'
    stripped = _strip_fences(fenced)
    parsed = json.loads(stripped)
    assert parsed == {"key": "value", "num": 1}, f"unexpected: {parsed}"
    print(f"✅ Fence stripped correctly: {parsed}")

    print()

    # Test 3: deliberate bad JSON → LLMError
    print("--- Test 3: bad JSON detection ---")
    try:
        json.loads("not json at all")
    except json.JSONDecodeError:
        print("✅ json.JSONDecodeError correctly raised on bad input")
