"""
llm_client.py — Unified LLM client for OpenAI, DeepSeek, Gemini, and Anthropic.

Provider is auto-detected from the model name prefix:
  gpt-* / o1-* / o3-* / o4-*  → OpenAI        (OPENAI_API_KEY)
  claude-*                     → Anthropic      (ANTHROPIC_API_KEY)
  gemini-*                     → Google Gemini  (GEMINI_API_KEY)
  deepseek-*                   → DeepSeek       (DEEPSEEK_API_KEY)

All providers require the appropriate API key set as an environment variable.
"""

import json
import os

from openai import OpenAI


# ── Provider detection ────────────────────────────────────────────────────────

def get_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("deepseek-"):
        return "deepseek"
    return "openai"


# ── Client factories ──────────────────────────────────────────────────────────

def get_openai_compat_client(model: str) -> OpenAI:
    """Return an OpenAI SDK client pointed at the correct base URL for the model."""
    provider = get_provider(model)
    if provider == "deepseek":
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
        )
    if provider == "gemini":
        return OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    # Default: OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_anthropic_client():
    """Return an Anthropic SDK client. Raises if the package is not installed."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The 'anthropic' package is required for Claude models. "
            "Install it with: pip install anthropic"
        ) from exc
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


# ── Unified text completion ───────────────────────────────────────────────────

def text_complete(
    model: str,
    messages: list[dict],
    temperature: float = 0,
    json_mode: bool = False,
) -> tuple[str, int]:
    """
    Send a chat completion and return (response_text, tokens_used).

    Works across all supported providers. When json_mode=True the response
    is requested as strict JSON (falls back to text parsing for Anthropic).
    """
    provider = get_provider(model)

    if provider == "anthropic":
        return _anthropic_text_complete(model, messages, temperature)

    # ── OpenAI-compatible path (openai, deepseek, gemini) ─────────────────────
    client = get_openai_compat_client(model)
    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    # o1/o3 reasoning models do not accept temperature != 1 in some versions
    is_reasoning = model.startswith(("o1", "o3", "o4"))
    create_kwargs: dict = {"model": model, "messages": messages, **kwargs}
    if not is_reasoning:
        create_kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**create_kwargs)
    content = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return content, tokens


def _anthropic_text_complete(
    model: str, messages: list[dict], temperature: float
) -> tuple[str, int]:
    """Anthropic SDK text completion (no tools)."""
    client = get_anthropic_client()

    # Separate system message from the rest
    system = None
    filtered: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            filtered.append(m)

    kwargs: dict = {}
    if system:
        kwargs["system"] = system

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        messages=filtered,
        **kwargs,
    )

    content = ""
    for block in resp.content:
        if hasattr(block, "text"):
            content = block.text
            break

    tokens = (
        (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0
    )
    return content, tokens


# ── Anthropic tool-call agent utilities ───────────────────────────────────────

def openai_tools_to_anthropic(openai_tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool specs to Anthropic format."""
    result = []
    for t in openai_tools:
        fn = t.get("function", {})
        result.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return result


def anthropic_content_to_dicts(content_blocks) -> list[dict]:
    """Serialize Anthropic content blocks to plain dicts for message history."""
    result = []
    for block in content_blocks:
        if block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return result
