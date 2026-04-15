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


# ── Seed data (sourced from Capstone_Final.xlsx) ──────────────────────────────
SEED_TEST_CASES = [
    # ── Conversational ────────────────────────────────────────────────────────
    {"category": "conversational", "question": "How many accounts exist in each office location?"},
    {"category": "conversational", "question": "What are the top 5 accounts with the highest revenue per employee?"},
    {"category": "conversational", "question": "Which sectors have the highest average revenue?"},
    {"category": "conversational", "question": "What are the 5 oldest companies in the database?"},
    {"category": "conversational", "question": "Which accounts have both high revenue and high propensity to buy?"},
    {"category": "conversational", "question": "How many accounts exist in each sector?"},
    {"category": "conversational", "question": "What are the top 5 accounts with highest propensity to buy?"},
    {"category": "conversational", "question": "Which products are the most expensive?"},
    {"category": "conversational", "question": "How many products exist in each series?"},
    {"category": "conversational", "question": "What is the average price of products in each series?"},
    {"category": "conversational", "question": "How many sales opportunities exist in each deal stage?"},
    {"category": "conversational", "question": "How many deals are in each deal stage?"},
    {"category": "conversational", "question": "What is the total value of all won deals?"},
    {"category": "conversational", "question": "Which sales agents closed the most deals?"},
    {"category": "conversational", "question": "What are the top 5 largest deals by value?"},
    {"category": "conversational", "question": "Which accounts generated the most revenue from won deals?"},
    {"category": "conversational", "question": "What is the average deal value for each sales agent?"},
    {"category": "conversational", "question": "Which sales agents generated the highest total revenue?"},
    {"category": "conversational", "question": "Which product generated the most revenue?"},
    {"category": "conversational", "question": "What is the total revenue generated per product series?"},
    {"category": "conversational", "question": "Which sales manager generated the most revenue through their team?"},
    {"category": "conversational", "question": "Which regional office generated the most revenue?"},
    {"category": "conversational", "question": "Which accounts have the highest total interaction count?"},
    {"category": "conversational", "question": "Which accounts have won deals but no recorded interactions?"},
    {"category": "conversational", "question": "Which sales agents have the highest win rate?"},
    {"category": "conversational", "question": "I have availability in my schedule, can you provide me with 3 recommendations for who to contact and why?"},
    {"category": "conversational", "question": "Which accounts should I prioritize this week to maximize potential revenue?"},
    {"category": "conversational", "question": "Which deals appear most promising right now and why?"},
    {"category": "conversational", "question": "Which deals should I focus on closing this quarter?"},
    {"category": "conversational", "question": "Which accounts might require proactive outreach soon?"},
    {"category": "conversational", "question": "Based on current data, what sales strategy would you recommend for the next month?"},
    {"category": "conversational", "question": "Are there any meetings, tasks, or open engagements that I should follow-up on?"},
    {"category": "conversational", "question": "Which accounts have not been contacted recently but have active opportunities?"},
    {"category": "conversational", "question": "Are there any deals that appear to be stalled in the pipeline?"},
    {"category": "conversational", "question": "Which accounts should receive follow-up before the end of the quarter?"},
    {"category": "conversational", "question": "Which opportunities have not progressed recently and may require attention?"},
    {"category": "conversational", "question": "Can you provide me with a summary of previous interactions with Yearin?"},
    {"category": "conversational", "question": "Summarize the recent sales activity for Account Yearin."},
    {"category": "conversational", "question": "Provide a short overview of our relationship with Account Yearin."},
    {"category": "conversational", "question": "What are the key takeaways from recent interactions with Account Yearin?"},
    {"category": "conversational", "question": "Provide a brief summary of the last three interactions with Account Yearin."},
    {"category": "conversational", "question": "What is the current status of the opportunities associated with Account Yearin?"},
    {"category": "conversational", "question": "Which customers have high propensity to buy but limited recent engagement?"},
    {"category": "conversational", "question": "Which accounts have frequent interactions but no closed deals yet?"},
    {"category": "conversational", "question": "Which sectors appear to generate the most successful deals?"},
    {"category": "conversational", "question": "What insights can you provide about recent sales activity across the pipeline?"},
    {"category": "conversational", "question": "Which accounts show strong engagement but low conversion rates?"},
    {"category": "conversational", "question": "Which sales activities appear to be most effective based on recent data?"},
    {"category": "conversational", "question": "What patterns do you observe among recently won deals?"},
    {"category": "conversational", "question": "Which accounts should I prepare for upcoming meetings with?"},
    {"category": "conversational", "question": "Which customer is closest to me?"},

    # ── SQL ───────────────────────────────────────────────────────────────────
    {
        "category": "sql",
        "question": "How many accounts exist in each office location?",
        "golden_sql": (
            "SELECT office_location, COUNT(*) AS total_accounts\n"
            "FROM accounts\n"
            "GROUP BY office_location\n"
            "ORDER BY total_accounts DESC"
        ),
    },
    {
        "category": "sql",
        "question": "What are the top 5 accounts with the highest revenue per employee?",
        "golden_sql": (
            "SELECT account, revenue / employees AS revenue_per_employee\n"
            "FROM accounts\n"
            "WHERE employees > 0\n"
            "ORDER BY revenue_per_employee DESC\n"
            "LIMIT 5;"
        ),
    },
    {
        "category": "sql",
        "question": "Which sectors have the highest average revenue?",
        "golden_sql": (
            "SELECT sector, AVG(revenue) AS avg_revenue\n"
            "FROM accounts\n"
            "GROUP BY sector\n"
            "ORDER BY avg_revenue DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "What are the 5 oldest companies in the database?",
        "golden_sql": (
            "SELECT account, year_established\n"
            "FROM accounts\n"
            "ORDER BY year_established ASC\n"
            "LIMIT 5;"
        ),
    },
    {
        "category": "sql",
        "question": "Which accounts have both high revenue and high propensity to buy?",
        "golden_sql": (
            "SELECT account, revenue, propensity_to_buy\n"
            "FROM accounts\n"
            "WHERE revenue > 3000\n"
            "AND propensity_to_buy > 0.65\n"
            "ORDER BY revenue DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "How many accounts exist in each sector?",
        "golden_sql": (
            "SELECT sector, COUNT(*) AS total_accounts\n"
            "FROM accounts\n"
            "GROUP BY sector\n"
            "ORDER BY total_accounts DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "What are the top 5 accounts with highest propensity to buy?",
        "golden_sql": (
            "SELECT account, propensity_to_buy\n"
            "FROM accounts\n"
            "ORDER BY propensity_to_buy DESC\n"
            "LIMIT 5;"
        ),
    },
    {
        "category": "sql",
        "question": "Which products are the most expensive?",
        "golden_sql": (
            "SELECT product, sales_price\n"
            "FROM products\n"
            "ORDER BY sales_price DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "How many products exist in each series?",
        "golden_sql": (
            "SELECT series, COUNT(*) AS total_products\n"
            "FROM products\n"
            "GROUP BY series\n"
            "ORDER BY total_products DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "What is the average price of products in each series?",
        "golden_sql": (
            "SELECT series, AVG(sales_price) AS avg_price\n"
            "FROM products\n"
            "GROUP BY series;"
        ),
    },
    {
        "category": "sql",
        "question": "How many sales opportunities exist in each deal stage?",
        "golden_sql": (
            "SELECT deal_stage, COUNT(*) AS total_opportunities\n"
            "FROM sales_pipeline\n"
            "GROUP BY deal_stage\n"
            "ORDER BY total_opportunities DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "How many deals are in each deal stage?",
        "golden_sql": (
            "SELECT deal_stage, COUNT(*) AS total_deals\n"
            "FROM sales_pipeline\n"
            "GROUP BY deal_stage\n"
            "ORDER BY total_deals DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "What is the total value of all won deals?",
        "golden_sql": (
            "SELECT SUM(close_value) AS total_won_value\n"
            "FROM sales_pipeline\n"
            "WHERE deal_stage = 'Won';"
        ),
    },
    {
        "category": "sql",
        "question": "Which sales agents closed the most deals?",
        "golden_sql": (
            "SELECT sales_agent, COUNT(*) AS deals_closed\n"
            "FROM sales_pipeline\n"
            "WHERE deal_stage = 'Won'\n"
            "GROUP BY sales_agent\n"
            "ORDER BY deals_closed DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "What are the top 5 largest deals by value?",
        "golden_sql": (
            "SELECT opportunity_id, account, close_value\n"
            "FROM sales_pipeline\n"
            "ORDER BY close_value DESC\n"
            "LIMIT 5;"
        ),
    },
    {
        "category": "sql",
        "question": "Which accounts generated the most revenue from won deals?",
        "golden_sql": (
            "SELECT account, SUM(close_value) AS total_revenue\n"
            "FROM sales_pipeline\n"
            "WHERE deal_stage = 'Won'\n"
            "GROUP BY account\n"
            "ORDER BY total_revenue DESC\n"
            "LIMIT 10;"
        ),
    },
    {
        "category": "sql",
        "question": "What is the average deal value for each sales agent?",
        "golden_sql": (
            "SELECT sales_agent, AVG(close_value) AS avg_deal_value\n"
            "FROM sales_pipeline\n"
            "GROUP BY sales_agent\n"
            "ORDER BY avg_deal_value DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "Which sales agents generated the highest total revenue?",
        "golden_sql": (
            "SELECT sales_agent, SUM(close_value) AS total_revenue\n"
            "FROM sales_pipeline\n"
            "WHERE deal_stage = 'Won'\n"
            "GROUP BY sales_agent\n"
            "ORDER BY total_revenue DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "Which product generated the most revenue?",
        "golden_sql": (
            "SELECT product, SUM(close_value) AS total_revenue\n"
            "FROM sales_pipeline\n"
            "WHERE deal_stage = 'Won'\n"
            "GROUP BY product\n"
            "ORDER BY total_revenue DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "What is the total revenue generated per product series?",
        "golden_sql": (
            "SELECT p.series, SUM(sp.close_value) AS total_revenue\n"
            "FROM sales_pipeline sp\n"
            "JOIN products p ON sp.product_id = p.product_id\n"
            "WHERE sp.deal_stage = 'Won'\n"
            "GROUP BY p.series\n"
            "ORDER BY total_revenue DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "Which sales manager generated the most revenue through their team?",
        "golden_sql": (
            "SELECT st.manager, SUM(sp.close_value) AS total_revenue\n"
            "FROM sales_pipeline sp\n"
            "JOIN sales_teams st ON sp.sales_agent = st.sales_agent\n"
            "WHERE sp.deal_stage = 'Won'\n"
            "GROUP BY st.manager\n"
            "ORDER BY total_revenue DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "Which regional office generated the most revenue?",
        "golden_sql": (
            "SELECT st.regional_office, SUM(sp.close_value) AS total_revenue\n"
            "FROM sales_pipeline sp\n"
            "JOIN sales_teams st ON sp.sales_agent = st.sales_agent\n"
            "WHERE sp.deal_stage = 'Won'\n"
            "GROUP BY st.regional_office\n"
            "ORDER BY total_revenue DESC;"
        ),
    },
    {
        "category": "sql",
        "question": "Which accounts have the highest total interaction count?",
        "golden_sql": (
            "SELECT account_name, COUNT(*) AS interaction_count\n"
            "FROM interactions\n"
            "GROUP BY account_name\n"
            "ORDER BY interaction_count DESC\n"
            "LIMIT 10;"
        ),
    },
    {
        "category": "sql",
        "question": "Which accounts have the most interactions?",
        "golden_sql": (
            "SELECT account_name, COUNT(*) AS interaction_count\n"
            "FROM interactions\n"
            "GROUP BY account_name\n"
            "ORDER BY interaction_count DESC\n"
            "LIMIT 10;"
        ),
    },
    {
        "category": "sql",
        "question": "Which sales agents have the highest win rate?",
        "golden_sql": (
            "SELECT\n"
            "  sales_agent,\n"
            "  SUM(CASE WHEN deal_stage = 'Won' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate\n"
            "FROM sales_pipeline\n"
            "GROUP BY sales_agent\n"
            "ORDER BY win_rate DESC;"
        ),
    },

    # ── Performance ───────────────────────────────────────────────────────────
    {"category": "performance", "question": "How many accounts exist in each office location?",
     "expected_rows": None, "time_threshold_ms": 10000.0},
    {"category": "performance", "question": "What are the top 5 accounts with the highest revenue per employee?",
     "expected_rows": 5, "time_threshold_ms": 6984.0},
    {"category": "performance", "question": "Which sectors have the highest average revenue?",
     "expected_rows": 10, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "What are the 5 oldest companies in the database?",
     "expected_rows": 5, "time_threshold_ms": 6131.0},
    {"category": "performance", "question": "Which accounts have both high revenue and high propensity to buy?",
     "expected_rows": 6, "time_threshold_ms": 7943.0},
    {"category": "performance", "question": "How many accounts exist in each sector?",
     "expected_rows": 10, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "What are the top 5 accounts with highest propensity to buy?",
     "expected_rows": 5, "time_threshold_ms": 6510.0},
    {"category": "performance", "question": "Which products are the most expensive?",
     "expected_rows": 7, "time_threshold_ms": 5808.0},
    {"category": "performance", "question": "How many products exist in each series?",
     "expected_rows": 3, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "What is the average price of products in each series?",
     "expected_rows": 3, "time_threshold_ms": 5680.0},
    {"category": "performance", "question": "How many sales opportunities exist in each deal stage?",
     "expected_rows": 4, "time_threshold_ms": 6401.0},
    {"category": "performance", "question": "How many deals are in each deal stage?",
     "expected_rows": 4, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "What is the total value of all won deals?",
     "expected_rows": 1, "time_threshold_ms": 9301.0},
    {"category": "performance", "question": "Which sales agents closed the most deals?",
     "expected_rows": 30, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "What are the top 5 largest deals by value?",
     "expected_rows": 5, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "Which accounts generated the most revenue from won deals?",
     "expected_rows": 85, "time_threshold_ms": 7411.0},
    {"category": "performance", "question": "What is the average deal value for each sales agent?",
     "expected_rows": 30, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "Which sales agents generated the highest total revenue?",
     "expected_rows": 30, "time_threshold_ms": 5957.0},
    {"category": "performance", "question": "Which product generated the most revenue?",
     "expected_rows": 1, "time_threshold_ms": 15634.0},
    {"category": "performance", "question": "What is the total revenue generated per product series?",
     "expected_rows": 3, "time_threshold_ms": 5925.0},
    {"category": "performance", "question": "Which sales manager generated the most revenue through their team?",
     "expected_rows": 1, "time_threshold_ms": 5347.0},
    {"category": "performance", "question": "Which regional office generated the most revenue?",
     "expected_rows": 0, "time_threshold_ms": 6344.0},
    {"category": "performance", "question": "Which accounts have the highest total interaction count?",
     "expected_rows": 85, "time_threshold_ms": 5027.0},
    {"category": "performance", "question": "Which accounts have won deals but no recorded interactions?",
     "expected_rows": 0, "time_threshold_ms": 5000.0},
    {"category": "performance", "question": "Which sales agents have the highest win rate?",
     "expected_rows": 30, "time_threshold_ms": 5000.0},
]


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

        # Seed test cases if none exist yet
        tc_count = con.execute("SELECT COUNT(*) FROM test_cases").fetchone()[0]
        if tc_count == 0:
            for i, tc in enumerate(SEED_TEST_CASES, start=1):
                con.execute("""
                    INSERT INTO test_cases
                        (id, category, question, golden_sql, expected_rows, time_threshold_ms)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    i,
                    tc["category"],
                    tc["question"],
                    tc.get("golden_sql"),
                    tc.get("expected_rows"),
                    tc.get("time_threshold_ms"),
                ])

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
