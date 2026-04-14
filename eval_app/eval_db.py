"""
eval_db.py — DuckDB-backed storage for test cases, prompts, runs, and results.
Uses eval.duckdb, completely separate from the sales.duckdb app database.
"""

import duckdb
import os
import json
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "eval.duckdb")

DEFAULT_SYSTEM_PROMPT = """You are a helpful sales assistant with access to a CRM database.

Current User: Evaluator

You have multiple tools available:
- text_to_sql: For flexible, ad-hoc queries about any data in the database
- open_work: For quickly getting outstanding work items (automatically filtered for current user)

IMPORTANT: For questions asking about multiple things (like "open work AND deals closing soon"):
1. Call open_work first
2. Then call text_to_sql for the additional information
3. After gathering all information, provide a synthesized, prioritized answer combining both results

Do NOT just return raw tool output - always provide a final synthesized answer after gathering information."""

DEFAULT_SQL_PROMPT = """You are a SQL expert. Given this database schema and a user question, generate a valid DuckDB SQL query.

{schema}

{context}

User question: {question}

Generate ONLY the SQL query, no explanation. Use read-only SELECT statements only.
Prefer using the views when appropriate for the question."""


def get_connection(read_only: bool = False):
    # Always open read-write; DuckDB raises ConnectionException if you mix
    # read_only=True and read_only=False connections to the same file.
    return duckdb.connect(DB_PATH)


def init_db():
    """Create all tables if they don't exist and seed default prompt."""
    con = get_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS test_cases (
                id          INTEGER PRIMARY KEY,
                category    VARCHAR NOT NULL,
                question    VARCHAR NOT NULL,
                golden_sql  VARCHAR,
                expected_output    VARCHAR,
                expected_rows      INTEGER,
                time_threshold_ms  DOUBLE,
                created_at  TIMESTAMP DEFAULT current_timestamp,
                updated_at  TIMESTAMP DEFAULT current_timestamp
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id           INTEGER PRIMARY KEY,
                name         VARCHAR NOT NULL,
                description  VARCHAR,
                system_prompt VARCHAR NOT NULL,
                sql_prompt   VARCHAR NOT NULL,
                is_default   BOOLEAN DEFAULT FALSE,
                created_at   TIMESTAMP DEFAULT current_timestamp,
                updated_at   TIMESTAMP DEFAULT current_timestamp
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id           INTEGER PRIMARY KEY,
                run_name     VARCHAR NOT NULL,
                model        VARCHAR NOT NULL,
                prompt_id    INTEGER,
                prompt_name  VARCHAR,
                started_at   TIMESTAMP DEFAULT current_timestamp,
                completed_at TIMESTAMP,
                total_cases  INTEGER DEFAULT 0,
                passed       INTEGER DEFAULT 0,
                failed       INTEGER DEFAULT 0
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id              INTEGER PRIMARY KEY,
                run_id          INTEGER NOT NULL,
                test_case_id    INTEGER,
                category        VARCHAR NOT NULL,
                question        VARCHAR NOT NULL,
                llm_response    VARCHAR,
                generated_sql   VARCHAR,
                passed          BOOLEAN,
                score           DOUBLE,
                error_message   VARCHAR,
                relevance       DOUBLE,
                accuracy        DOUBLE,
                completeness    DOUBLE,
                actionability   DOUBLE,
                safety          DOUBLE,
                llm_latency_ms  DOUBLE,
                execution_ms    DOUBLE,
                total_time_ms   DOUBLE,
                tokens_used     INTEGER,
                rows_returned   INTEGER,
                created_at      TIMESTAMP DEFAULT current_timestamp
            )
        """)

        # Seed the default prompt if no prompts exist
        count = con.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
        if count == 0:
            con.execute("""
                INSERT INTO prompts (id, name, description, system_prompt, sql_prompt, is_default)
                VALUES (1, 'Default Prompt', 'Original system prompt from the reference chatbot',
                        ?, ?, TRUE)
            """, [DEFAULT_SYSTEM_PROMPT, DEFAULT_SQL_PROMPT])

        con.commit()
    finally:
        con.close()


# ── Test Cases ────────────────────────────────────────────────────────────────

def get_test_cases(category: Optional[str] = None) -> list[dict]:
    con = get_connection(read_only=True)
    try:
        if category:
            rows = con.execute(
                "SELECT * FROM test_cases WHERE category = ? ORDER BY id",
                [category]
            ).fetchdf()
        else:
            rows = con.execute("SELECT * FROM test_cases ORDER BY id").fetchdf()
        return rows.to_dict("records")
    finally:
        con.close()


def add_test_case(category: str, question: str, golden_sql: str = None,
                  expected_output: str = None, expected_rows: int = None,
                  time_threshold_ms: float = None) -> int:
    con = get_connection()
    try:
        next_id = (con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM test_cases").fetchone()[0])
        con.execute("""
            INSERT INTO test_cases (id, category, question, golden_sql, expected_output,
                                    expected_rows, time_threshold_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [next_id, category, question, golden_sql, expected_output,
              expected_rows, time_threshold_ms])
        con.commit()
        return next_id
    finally:
        con.close()


def update_test_case(tc_id: int, **kwargs):
    allowed = {"question", "golden_sql", "expected_output", "expected_rows", "time_threshold_ms"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    updates["updated_at"] = datetime.now()
    con = get_connection()
    try:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        con.execute(
            f"UPDATE test_cases SET {set_clause} WHERE id = ?",
            list(updates.values()) + [tc_id]
        )
        con.commit()
    finally:
        con.close()


def delete_test_case(tc_id: int):
    con = get_connection()
    try:
        con.execute("DELETE FROM test_cases WHERE id = ?", [tc_id])
        con.commit()
    finally:
        con.close()


def count_test_cases() -> int:
    con = get_connection(read_only=True)
    try:
        return con.execute("SELECT COUNT(*) FROM test_cases").fetchone()[0]
    finally:
        con.close()


# ── Prompts ───────────────────────────────────────────────────────────────────

def get_prompts() -> list[dict]:
    con = get_connection(read_only=True)
    try:
        return con.execute("SELECT * FROM prompts ORDER BY id").fetchdf().to_dict("records")
    finally:
        con.close()


def get_prompt(prompt_id: int) -> Optional[dict]:
    con = get_connection(read_only=True)
    try:
        rows = con.execute(
            "SELECT * FROM prompts WHERE id = ?", [prompt_id]
        ).fetchdf()
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()
    finally:
        con.close()


def add_prompt(name: str, description: str, system_prompt: str, sql_prompt: str) -> int:
    con = get_connection()
    try:
        next_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM prompts").fetchone()[0]
        con.execute("""
            INSERT INTO prompts (id, name, description, system_prompt, sql_prompt, is_default)
            VALUES (?, ?, ?, ?, ?, FALSE)
        """, [next_id, name, description, system_prompt, sql_prompt])
        con.commit()
        return next_id
    finally:
        con.close()


def update_prompt(prompt_id: int, name: str, description: str,
                  system_prompt: str, sql_prompt: str):
    con = get_connection()
    try:
        con.execute("""
            UPDATE prompts SET name=?, description=?, system_prompt=?, sql_prompt=?,
                               updated_at=current_timestamp
            WHERE id = ?
        """, [name, description, system_prompt, sql_prompt, prompt_id])
        con.commit()
    finally:
        con.close()


def delete_prompt(prompt_id: int):
    con = get_connection()
    try:
        # Don't delete if it's the only prompt
        count = con.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
        if count <= 1:
            raise ValueError("Cannot delete the last remaining prompt.")
        con.execute("DELETE FROM prompts WHERE id = ?", [prompt_id])
        # If we deleted the default, set another one as default
        default_count = con.execute(
            "SELECT COUNT(*) FROM prompts WHERE is_default = TRUE"
        ).fetchone()[0]
        if default_count == 0:
            first_id = con.execute("SELECT MIN(id) FROM prompts").fetchone()[0]
            con.execute("UPDATE prompts SET is_default = TRUE WHERE id = ?", [first_id])
        con.commit()
    finally:
        con.close()


def set_default_prompt(prompt_id: int):
    con = get_connection()
    try:
        con.execute("UPDATE prompts SET is_default = FALSE")
        con.execute("UPDATE prompts SET is_default = TRUE WHERE id = ?", [prompt_id])
        con.commit()
    finally:
        con.close()


# ── Runs & Results ────────────────────────────────────────────────────────────

def create_run(run_name: str, model: str, prompt_id: int, prompt_name: str) -> int:
    con = get_connection()
    try:
        next_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM eval_runs").fetchone()[0]
        con.execute("""
            INSERT INTO eval_runs (id, run_name, model, prompt_id, prompt_name)
            VALUES (?, ?, ?, ?, ?)
        """, [next_id, run_name, model, prompt_id, prompt_name])
        con.commit()
        return next_id
    finally:
        con.close()


def finalize_run(run_id: int, total: int, passed: int, failed: int):
    con = get_connection()
    try:
        con.execute("""
            UPDATE eval_runs
            SET completed_at = current_timestamp, total_cases = ?, passed = ?, failed = ?
            WHERE id = ?
        """, [total, passed, failed, run_id])
        con.commit()
    finally:
        con.close()


def save_result(run_id: int, test_case_id: Optional[int], category: str,
                question: str, result: dict):
    con = get_connection()
    try:
        next_id = con.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM eval_results"
        ).fetchone()[0]
        con.execute("""
            INSERT INTO eval_results (
                id, run_id, test_case_id, category, question,
                llm_response, generated_sql, passed, score, error_message,
                relevance, accuracy, completeness, actionability, safety,
                llm_latency_ms, execution_ms, total_time_ms, tokens_used, rows_returned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            next_id, run_id, test_case_id, category, question,
            result.get("llm_response"), result.get("generated_sql"),
            result.get("passed"), result.get("score"), result.get("error_message"),
            result.get("relevance"), result.get("accuracy"),
            result.get("completeness"), result.get("actionability"), result.get("safety"),
            result.get("llm_latency_ms"), result.get("execution_ms"),
            result.get("total_time_ms"), result.get("tokens_used"), result.get("rows_returned"),
        ])
        con.commit()
    finally:
        con.close()


def get_runs() -> list[dict]:
    con = get_connection(read_only=True)
    try:
        df = con.execute("""
            SELECT id, run_name, model, prompt_name, started_at, completed_at,
                   total_cases, passed, failed
            FROM eval_runs
            ORDER BY id DESC
        """).fetchdf()
        return df.to_dict("records")
    finally:
        con.close()


def get_results(run_id: int) -> list[dict]:
    con = get_connection(read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY id",
            [run_id]
        ).fetchdf()
        return df.to_dict("records")
    finally:
        con.close()


def get_results_for_runs(run_ids: list[int]) -> list[dict]:
    con = get_connection(read_only=True)
    try:
        placeholders = ", ".join("?" * len(run_ids))
        df = con.execute(
            f"SELECT * FROM eval_results WHERE run_id IN ({placeholders}) ORDER BY run_id, id",
            run_ids
        ).fetchdf()
        return df.to_dict("records")
    finally:
        con.close()
