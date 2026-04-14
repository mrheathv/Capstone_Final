"""
importer.py — Import test cases from Capstone_Final.xlsx into eval.duckdb.
Safe to call multiple times; skips import if test_cases table already has data.
"""

import re
import json
import os
import openpyxl

EXCEL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Reference Files", "rag_salesbot-main", "Capstone_Final.xlsx"
)

DEFAULT_TIME_THRESHOLD_MS = 10000.0  # 10 seconds generous default


def _extract_json(text: str) -> dict:
    """
    Extract and parse the first JSON object found in a string.
    Handles Excel cell-merge artifacts (e.g. stray 'M12' tokens).
    """
    if not text:
        return {}
    # Find the outermost {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    json_str = text[start: end + 1]
    # Remove stray Excel merge tokens like ,M12 or ,AB3 between a number and whitespace
    json_str = re.sub(r",([A-Z]+\d+)\b", "", json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {}


def import_from_excel(db) -> dict:
    """
    Read Capstone_Final.xlsx and insert test cases into the database.

    Args:
        db: the eval_db module (passed in to avoid circular import issues)

    Returns:
        dict with counts of inserted records per category.
    """
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    wb = openpyxl.load_workbook(EXCEL_PATH)
    counts = {"conversational": 0, "sql": 0, "performance": 0}

    # ── Conversational ────────────────────────────────────────────────────────
    ws = wb["Conversational"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        q_no, question, q_type, eval_purpose, expected_output, *_ = list(row) + [None] * 8
        if not question:
            continue
        db.add_test_case(
            category="conversational",
            question=str(question).strip(),
            expected_output=str(expected_output).strip() if expected_output else None,
        )
        counts["conversational"] += 1

    # ── SQL ───────────────────────────────────────────────────────────────────
    ws = wb["SQL"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        question, golden_sql, *_ = list(row) + [None] * 4
        if not question:
            continue
        db.add_test_case(
            category="sql",
            question=str(question).strip(),
            golden_sql=str(golden_sql).strip() if golden_sql else None,
        )
        counts["sql"] += 1

    # ── Performance ───────────────────────────────────────────────────────────
    ws = wb["Performance"]
    for row in ws.iter_rows(min_row=2, values_only=True):
        question, output_text, *_ = list(row) + [None] * 4
        if not question:
            continue

        # Try to extract metrics from the stored output JSON
        metrics = _extract_json(str(output_text) if output_text else "")
        rows_returned = None
        time_threshold_ms = DEFAULT_TIME_THRESHOLD_MS

        if metrics:
            execution = metrics.get("execution", {})
            rows_returned = execution.get("rows_returned")

            generation = metrics.get("generation", {})
            total_ms = generation.get("total_ms")
            if total_ms:
                # Use 2× the recorded time as the threshold, minimum 5 s
                time_threshold_ms = max(float(total_ms) * 2, 5000.0)

        db.add_test_case(
            category="performance",
            question=str(question).strip(),
            expected_rows=int(rows_returned) if rows_returned is not None else None,
            time_threshold_ms=time_threshold_ms,
        )
        counts["performance"] += 1

    return counts
