"""
judge.py — LLM-as-judge for scoring conversational responses.

The scoring rubric (dimensions, weights, descriptions) is passed in at call
time from eval_db.get_rubric("conversational"), so every change made in the
UI is immediately reflected in what the judge is asked to evaluate.

Supports any model/provider via llm_client (OpenAI, Anthropic, Gemini, DeepSeek).
"""

import json

from llm_client import get_provider, text_complete

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI sales assistant responses.
You must be objective and base scores strictly on the criteria provided.
Always respond with valid JSON only."""


# ── Dynamic prompt builder ────────────────────────────────────────────────────

def _build_judge_prompt(question: str, response: str, rubric: list[dict]) -> str:
    """
    Build the judge user prompt from a live rubric (list of dicts from DB).
    Only items with a non-null weight are treated as scoring dimensions.
    Weights are normalized to sum to 1.0 in case the user edited them.
    """
    dims = [r for r in rubric if r.get("weight") is not None]
    total_weight = sum(float(r["weight"]) for r in dims) or 1.0

    rubric_lines = []
    for r in dims:
        norm_w = float(r["weight"]) / total_weight
        pct = round(norm_w * 100)
        rubric_lines.append(
            f"- {r['item_key']} (weight {pct}%): Score 1–5. {r['description']}"
        )

    rubric_text = "\n".join(rubric_lines)
    dim_keys = [r["item_key"] for r in dims]
    json_fields = ", ".join(f'"{k}": <1-5>' for k in dim_keys)

    return (
        f"Evaluate the following AI sales assistant response on "
        f"{len(dims)} dimension(s), each scored 1–5.\n\n"
        f"QUESTION ASKED:\n{question}\n\n"
        f"AI RESPONSE:\n{response}\n\n"
        f"SCORING RUBRIC:\n{rubric_text}\n\n"
        f"Respond with ONLY this JSON (no extra text):\n"
        f"{{{json_fields}, \"reasoning\": \"<one sentence explaining the overall score>\"}}"
    )


def _weights_from_rubric(rubric: list[dict]) -> dict[str, float]:
    """Extract normalized weights dict {item_key: float} from rubric rows."""
    dims = [r for r in rubric if r.get("weight") is not None]
    total = sum(float(r["weight"]) for r in dims) or 1.0
    return {r["item_key"]: float(r["weight"]) / total for r in dims}


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """
    Parse JSON from a model response, handling markdown fences and prose wrapping.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:])
    if text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[:-1])
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from judge response: {text[:300]}")


# ── Main scoring function ─────────────────────────────────────────────────────

def score_response(question: str, response: str, model: str = "gpt-4o-mini",
                   rubric: list[dict] | None = None) -> dict:
    """
    Score an AI response using an LLM-as-judge.

    rubric   — list of dicts from eval_db.get_rubric("conversational").
               When provided the judge prompt and weights are derived from it,
               so edits made in the UI are immediately in effect.
               Falls back to a minimal default when None.

    Returns a dict with one key per scoring dimension plus:
      weighted_score, reasoning, error
    Pass threshold: weighted_score >= 3.0
    """
    # Build prompt + weights from live rubric (or a minimal fallback)
    if rubric:
        user_content = _build_judge_prompt(question, response, rubric)
        weights = _weights_from_rubric(rubric)
    else:
        # Fallback — basic single-dimension scoring so the app doesn't crash
        weights = {"relevance": 1.0}
        user_content = (
            f"Rate this sales assistant response on relevance (1–5).\n\n"
            f"QUESTION: {question}\n\nRESPONSE: {response}\n\n"
            '{"relevance": <1-5>, "reasoning": "<one sentence>"}'
        )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        provider = get_provider(model)
        use_json_mode = provider != "anthropic"

        content, _ = text_complete(model, messages, temperature=0, json_mode=use_json_mode)
        scores = _parse_json(content)

        # Clamp each scored dimension to [1, 5]
        for dim in weights:
            scores[dim] = max(1.0, min(5.0, float(scores.get(dim, 1))))

        weighted = sum(scores[dim] * w for dim, w in weights.items())
        scores["weighted_score"] = round(weighted, 3)
        scores["error"] = None
        return scores

    except Exception as exc:
        empty = {dim: None for dim in weights}
        return {
            **empty,
            "weighted_score": None,
            "reasoning": None,
            "error": str(exc),
        }


def weighted_score(scores: dict, rubric: list[dict] | None = None) -> float:
    """Compute weighted score. Uses rubric weights when provided."""
    weights = _weights_from_rubric(rubric) if rubric else {"relevance": 1.0}
    try:
        return sum(float(scores[dim]) * w for dim, w in weights.items())
    except (KeyError, TypeError):
        return 0.0
