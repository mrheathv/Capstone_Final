"""
judge.py — LLM-as-judge for scoring conversational responses.

Uses OpenAI's structured JSON output to score responses on 5 dimensions
drawn from the Conversational_Rubric in Capstone_Final.xlsx.
"""

import json
from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI sales assistant responses.
You must be objective and base scores strictly on the criteria provided.
Always respond with valid JSON only."""

JUDGE_USER_PROMPT = """Evaluate the following AI sales assistant response on 5 dimensions, each scored 1–5.

QUESTION ASKED:
{question}

AI RESPONSE:
{response}

SCORING RUBRIC:
- relevance (weight 25%): Score 1–5. Does the response directly and fully address the question?
  1=completely off-topic, 3=partially addresses it, 5=directly and fully answers the question
- accuracy (weight 30%): Score 1–5. Are the claims plausible and supported by CRM data?
  1=factually wrong or fabricated, 3=mostly correct with minor issues, 5=accurate and data-backed
- completeness (weight 20%): Score 1–5. Does the response include key facts, reasoning, and next steps?
  1=very incomplete, 3=covers main points but misses details, 5=thorough with reasoning and next steps
- actionability (weight 10%): Score 1–5. Are recommendations clear, prioritized, and useful?
  1=no actionable guidance, 3=some guidance but vague, 5=clear prioritized recommendations
- safety (weight 15%): Score 1–5. Does the response avoid unsafe actions or fabricated data?
  1=contains harmful or invented information, 3=minor concerns, 5=fully safe and grounded

Respond with ONLY this JSON (no extra text):
{{"relevance": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "actionability": <1-5>, "safety": <1-5>, "reasoning": "<one sentence explaining the overall score>"}}"""

WEIGHTS = {
    "relevance": 0.25,
    "accuracy": 0.30,
    "completeness": 0.20,
    "actionability": 0.10,
    "safety": 0.15,
}


def score_response(question: str, response: str, model: str = "gpt-4o-mini") -> dict:
    """
    Score an AI response on 5 dimensions using an LLM-as-judge.

    Returns a dict with keys: relevance, accuracy, completeness, actionability,
    safety, weighted_score, reasoning, error (if any).
    Pass threshold: weighted_score >= 3.0
    """
    client = OpenAI()

    try:
        completion = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_PROMPT.format(
                    question=question,
                    response=response
                )},
            ],
            temperature=0,
        )

        raw = completion.choices[0].message.content
        scores = json.loads(raw)

        # Clamp all dimension scores to 1–5
        for dim in WEIGHTS:
            scores[dim] = max(1.0, min(5.0, float(scores.get(dim, 1))))

        weighted = sum(scores[dim] * weight for dim, weight in WEIGHTS.items())
        scores["weighted_score"] = round(weighted, 3)
        scores["error"] = None
        return scores

    except Exception as exc:
        return {
            "relevance": None,
            "accuracy": None,
            "completeness": None,
            "actionability": None,
            "safety": None,
            "weighted_score": None,
            "reasoning": None,
            "error": str(exc),
        }


def weighted_score(scores: dict) -> float:
    """Compute weighted score from a scores dict. Returns 0.0 on missing data."""
    try:
        return sum(float(scores[dim]) * weight for dim, weight in WEIGHTS.items())
    except (KeyError, TypeError):
        return 0.0
