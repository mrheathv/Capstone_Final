"""
evaluator.py — Evaluation runners for all three test categories.

Reuses schema/context from the reference app but provides self-contained
versions of SQL generation and the agent loop that:
  - Accept any model from any supported provider (OpenAI, Anthropic, Gemini, DeepSeek)
  - Accept a custom prompt dict (system_prompt, sql_prompt)
  - Accept a separate judge_model for conversational scoring
  - Do not depend on streamlit session_state
"""

import os
import sys
import json
import time

# ── Reference-app imports ────────────────────────────────────────────────────
_REF_APP = os.path.join(
    os.path.dirname(__file__), "..", "Reference Files",
    "rag_salesbot-main", "app"
)
if _REF_APP not in sys.path:
    sys.path.insert(0, _REF_APP)

_ABS_DB_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "Reference Files",
    "rag_salesbot-main", "db", "sales.duckdb"
))

import database.connection as _conn
import database.schema as _schema
_conn.DB_PATH = _ABS_DB_PATH
_schema.DB_PATH = _ABS_DB_PATH

from database.connection import db_query
from database.schema import get_schema_info, get_business_context

import duckdb

import llm_client
from llm_client import get_provider, text_complete
import judge as _judge


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _clean_sql(raw: str) -> str:
    """Strip markdown fences from an LLM-generated SQL string."""
    sql = raw.strip()
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(lines[1:])
    if sql.endswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(lines[:-1])
    return sql.strip()


def _validate_sql(sql: str) -> tuple[bool, str]:
    upper = sql.upper().strip()
    if not upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed"
    for kw in ("DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"):
        if kw in upper:
            return False, f"Dangerous keyword detected: {kw}"
    return True, ""


def _dataframes_match(df_a, df_b, tolerance: float = 0.01) -> bool:
    """Return True if two DataFrames contain the same rows (order-insensitive)."""
    if df_a is None or df_b is None:
        return False
    if df_a.shape != df_b.shape:
        return False
    try:
        a = df_a.sort_values(by=list(df_a.columns)).reset_index(drop=True)
        b = df_b.sort_values(by=list(df_b.columns)).reset_index(drop=True)
        for col_a, col_b in zip(a.columns, b.columns):
            import pandas as pd
            series_a = a[col_a]
            series_b = b[col_b]
            if pd.api.types.is_numeric_dtype(series_a) and pd.api.types.is_numeric_dtype(series_b):
                if not ((series_a - series_b).abs() <= tolerance * series_a.abs().clip(lower=1)).all():
                    return False
            else:
                if not (series_a.astype(str) == series_b.astype(str)).all():
                    return False
        return True
    except Exception:
        return False


# ── SQL Generation (prompt-aware, multi-provider) ─────────────────────────────

def _generate_sql(question: str, model: str, sql_prompt_template: str,
                  max_attempts: int = 2) -> tuple[str, str, float, int]:
    """
    Generate SQL for *question* using *model* and the given prompt template.
    Supports all providers via llm_client.text_complete.

    Returns: (sql, error_message, llm_latency_ms, total_tokens)
    """
    schema = get_schema_info()
    context = get_business_context()

    last_error = ""
    last_sql = ""
    llm_latency_ms = 0.0
    total_tokens = 0

    for attempt in range(max_attempts):
        if attempt == 0:
            prompt = sql_prompt_template.format(
                schema=schema, context=context, question=question
            )
        else:
            prompt = (
                f"Your previous SQL query failed with this error:\n\nError: {last_error}\n\n"
                f"Previous query:\n{last_sql}\n\nSchema:\n{schema}\n\n"
                f"User question: {question}\n\n"
                "Please fix the query. Generate ONLY the corrected SQL, no explanation."
            )

        t0 = time.perf_counter()
        content, tokens = text_complete(
            model, [{"role": "user", "content": prompt}]
        )
        llm_latency_ms += (time.perf_counter() - t0) * 1000
        total_tokens += tokens

        sql = _clean_sql(content)
        valid, err = _validate_sql(sql)
        if not valid:
            last_error, last_sql = err, sql
            continue

        try:
            db_query(sql)  # test-execute
            return sql, "", llm_latency_ms, total_tokens
        except Exception as exc:
            last_error, last_sql = str(exc), sql

    return last_sql, last_error, llm_latency_ms, total_tokens


# ── Tool handlers ─────────────────────────────────────────────────────────────

def _open_work_standalone(args: dict) -> str:
    """Streamlit-free version of open_work_handler."""
    limit = args.get("limit", 25)
    sales_agent = args.get("sales_agent", "Evaluator")
    try:
        sql = f"""
            SELECT account_id, account AS account_name, deal_stage, sales_agent,
                   product, activity_type, status_lc, d_interaction AS last_activity_date,
                   comment
            FROM v_open_work
            WHERE LOWER(sales_agent) = LOWER('{sales_agent}')
            ORDER BY d_interaction DESC NULLS LAST
            LIMIT {limit}
        """
        df = db_query(sql)
        if df.empty:
            return f"No outstanding work items found for '{sales_agent}'."
        lines = [f"Outstanding Work Items ({len(df)} found):"]
        for _, row in df.iterrows():
            line = f"- {row.get('account_name', 'Unknown')} • {row.get('deal_stage', '')} • {row.get('product', '')}"
            comment = str(row.get("comment", "") or "")
            if comment.strip():
                lines.append(f"  {comment[:80]}{'...' if len(comment) > 80 else ''}")
            lines.append(line)
        return "\n".join(lines)
    except Exception as exc:
        return f"Error fetching open work: {exc}"


def _text_to_sql_standalone(args: dict, model: str, sql_prompt_template: str) -> str:
    """Streamlit-free version of text_to_sql_handler."""
    question = args.get("question", "")
    if not question:
        return "Error: No question provided."
    sql, error, _, _ = _generate_sql(question, model, sql_prompt_template)
    if error:
        return f"SQL generation failed: {error}\n\nLast SQL:\n```sql\n{sql}\n```"
    try:
        df = db_query(sql)
        if df.empty:
            return f"No results found.\n\nSQL:\n```sql\n{sql}\n```"
        return (
            f"**SQL Query:**\n```sql\n{sql}\n```\n\n"
            f"Found {len(df)} results:\n\n```\n{df.to_string(index=False)}\n```"
        )
    except Exception as exc:
        return f"Error executing query: {exc}\n\nSQL:\n```sql\n{sql}\n```"


# ── Tool spec (OpenAI format) ─────────────────────────────────────────────────

_TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "text_to_sql",
            "description": "Generate and execute a SQL query to answer questions about the sales CRM database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The natural-language question to answer with SQL"}
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_work",
            "description": "Get outstanding work items (deals in Engaging stage) for a sales agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sales_agent": {"type": "string", "description": "Sales agent name"},
                    "limit": {"type": "integer", "description": "Max rows to return"},
                },
            },
        },
    },
]


# ── Agent loops ───────────────────────────────────────────────────────────────

def _dispatch_tool(tool_name: str, tool_args: dict, model: str,
                   sql_prompt_template: str) -> str:
    if tool_name == "text_to_sql":
        return _text_to_sql_standalone(tool_args, model, sql_prompt_template)
    if tool_name == "open_work":
        return _open_work_standalone(tool_args)
    return f"Unknown tool: {tool_name}"


def _run_agent_openai_compat(question: str, system_prompt: str, model: str,
                              sql_prompt_template: str,
                              max_iterations: int = 5) -> tuple[str, float]:
    """ReAct agent loop for OpenAI-compatible providers (OpenAI, Deepseek, Gemini)."""
    client = llm_client.get_openai_compat_client(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    total_latency_ms = 0.0

    for _ in range(max_iterations):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=_TOOLS_SPEC,
            tool_choice="auto",
        )
        total_latency_ms += (time.perf_counter() - t0) * 1000

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "No answer generated.", total_latency_ms

        messages.append(msg)

        for tc in msg.tool_calls:
            result = _dispatch_tool(
                tc.function.name, json.loads(tc.function.arguments),
                model, sql_prompt_template,
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })

    return "Processing limit reached without a final answer.", total_latency_ms


def _run_agent_anthropic(question: str, system_prompt: str, model: str,
                         sql_prompt_template: str,
                         max_iterations: int = 5) -> tuple[str, float]:
    """ReAct agent loop for Anthropic Claude models."""
    client = llm_client.get_anthropic_client()
    anthropic_tools = llm_client.openai_tools_to_anthropic(_TOOLS_SPEC)

    messages: list[dict] = [{"role": "user", "content": question}]
    total_latency_ms = 0.0

    for _ in range(max_iterations):
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=anthropic_tools,
        )
        total_latency_ms += (time.perf_counter() - t0) * 1000

        # End turn — extract text answer
        if resp.stop_reason == "end_turn":
            for block in resp.content:
                if hasattr(block, "text"):
                    return block.text, total_latency_ms
            return "No answer generated.", total_latency_ms

        # Serialize assistant message and collect tool results
        messages.append({
            "role": "assistant",
            "content": llm_client.anthropic_content_to_dicts(resp.content),
        })

        tool_results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = _dispatch_tool(
                    block.name, block.input, model, sql_prompt_template
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return "Processing limit reached without a final answer.", total_latency_ms


def _run_agent(question: str, system_prompt: str, model: str,
               sql_prompt_template: str, max_iterations: int = 5) -> tuple[str, float]:
    """Dispatch to the correct agent loop based on provider."""
    if get_provider(model) == "anthropic":
        return _run_agent_anthropic(
            question, system_prompt, model, sql_prompt_template, max_iterations
        )
    return _run_agent_openai_compat(
        question, system_prompt, model, sql_prompt_template, max_iterations
    )


# ── Public evaluation functions ────────────────────────────────────────────────

def run_sql_test(question: str, golden_sql: str, model: str,
                 prompt: dict, expected_rows: int = None) -> dict:
    """
    Evaluate SQL generation for one test case.

    Pass criteria:
      - valid SELECT query generated
      - query executes without error
      - result matches golden_sql output (when golden_sql provided)
      - row count matches expected_rows (when expected_rows provided)

    Returns a result dict suitable for eval_db.save_result().
    """
    sql_prompt = prompt.get("sql_prompt", "")
    t_start = time.perf_counter()

    generated_sql, error, llm_latency_ms, tokens = _generate_sql(
        question, model, sql_prompt
    )

    valid, valid_err = (
        _validate_sql(generated_sql) if generated_sql else (False, "No SQL generated")
    )
    executed = False
    accurate = False
    result_df = None
    exec_ms = 0.0
    rows_returned = None
    exec_error = error or valid_err

    if valid and not error:
        try:
            t_exec = time.perf_counter()
            result_df = db_query(generated_sql)
            exec_ms = (time.perf_counter() - t_exec) * 1000
            executed = True
            rows_returned = len(result_df)
        except Exception as exc:
            exec_error = str(exc)

    if executed and golden_sql:
        try:
            golden_df = db_query(golden_sql.strip())
            accurate = _dataframes_match(result_df, golden_df)
        except Exception:
            accurate = False

    rows_ok = (rows_returned == expected_rows) if (executed and expected_rows is not None) else True

    total_time_ms = (time.perf_counter() - t_start) * 1000
    passed = valid and executed and (accurate if golden_sql else True) and rows_ok

    detail_parts = [f"Valid: {valid}", f"Executed: {executed}", f"Accurate: {accurate}"]
    if expected_rows is not None:
        detail_parts.append(f"Rows OK: {rows_ok} (got {rows_returned}, expected {expected_rows})")

    return {
        "passed": passed,
        "generated_sql": generated_sql,
        "llm_response": " | ".join(detail_parts),
        "error_message": exec_error if not passed else None,
        "llm_latency_ms": llm_latency_ms,
        "execution_ms": exec_ms,
        "total_time_ms": total_time_ms,
        "tokens_used": tokens,
        "rows_returned": rows_returned,
        "score": 1.0 if passed else 0.0,
    }


def run_conversational_test(question: str, model: str, prompt: dict,
                             judge_model: str = "gpt-4o-mini") -> dict:
    """
    Evaluate the full agent loop on one conversational question.
    Scores the response using LLM-as-judge with a separately configurable model.
    """
    system_prompt = prompt.get("system_prompt", "")
    sql_prompt = prompt.get("sql_prompt", "")

    t_start = time.perf_counter()
    answer, llm_latency_ms = _run_agent(question, system_prompt, model, sql_prompt)
    total_time_ms = (time.perf_counter() - t_start) * 1000

    scores = _judge.score_response(question, answer, judge_model)
    weighted = scores.get("weighted_score") or 0.0
    passed = (weighted >= 3.0) if scores.get("error") is None else False

    return {
        "passed": passed,
        "llm_response": answer,
        "generated_sql": None,
        "score": round(weighted, 3),
        "relevance": scores.get("relevance"),
        "accuracy": scores.get("accuracy"),
        "completeness": scores.get("completeness"),
        "actionability": scores.get("actionability"),
        "safety": scores.get("safety"),
        "error_message": scores.get("error"),
        "llm_latency_ms": llm_latency_ms,
        "execution_ms": None,
        "total_time_ms": total_time_ms,
        "tokens_used": None,
        "rows_returned": None,
    }


def run_performance_test(question: str, expected_rows: int,
                         time_threshold_ms: float, model: str,
                         prompt: dict) -> dict:
    """Evaluate SQL generation and measure performance metrics."""
    sql_prompt = prompt.get("sql_prompt", "")
    t_start = time.perf_counter()

    generated_sql, error, llm_latency_ms, tokens = _generate_sql(
        question, model, sql_prompt
    )

    valid, valid_err = (
        _validate_sql(generated_sql) if generated_sql else (False, "No SQL generated")
    )
    executed = False
    exec_ms = 0.0
    rows_returned = None
    exec_error = error or valid_err

    if valid and not error:
        try:
            t_exec = time.perf_counter()
            df = db_query(generated_sql)
            exec_ms = (time.perf_counter() - t_exec) * 1000
            executed = True
            rows_returned = len(df)
        except Exception as exc:
            exec_error = str(exc)

    total_time_ms = (time.perf_counter() - t_start) * 1000

    time_ok = total_time_ms <= (time_threshold_ms or float("inf"))
    rows_ok = (rows_returned == expected_rows) if expected_rows is not None else True
    passed = executed and time_ok and rows_ok

    detail_parts = [f"Executed: {executed}", f"Time OK: {time_ok} ({total_time_ms:.0f}ms)"]
    if expected_rows is not None:
        detail_parts.append(f"Rows OK: {rows_ok} (got {rows_returned}, expected {expected_rows})")

    return {
        "passed": passed,
        "generated_sql": generated_sql,
        "llm_response": " | ".join(detail_parts),
        "error_message": exec_error if not executed else None,
        "llm_latency_ms": llm_latency_ms,
        "execution_ms": exec_ms,
        "total_time_ms": total_time_ms,
        "tokens_used": tokens,
        "rows_returned": rows_returned,
        "score": 1.0 if passed else 0.0,
    }
