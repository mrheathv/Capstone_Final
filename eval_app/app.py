"""
app.py — Standalone LLM Evaluation App (Streamlit)

Run with:
    cd eval_app
    streamlit run app.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

import streamlit as st

# ── Bootstrap ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import eval_db as db
import evaluator

db.init_db()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Evaluation Suite",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 LLM Evaluation Suite")
st.caption("Evaluate RAG chatbot quality across Conversational, SQL, and Performance dimensions.")

AVAILABLE_MODELS = [
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4.1",            # Latest flagship; best instruction-following for RAG
    "gpt-4.1-mini",       # Fast & cost-efficient; strong RAG performance
    "gpt-4.1-nano",       # Fastest/cheapest; good for high-volume eval runs
    "gpt-4o",             # Multimodal flagship; excellent context synthesis
    "gpt-4o-mini",        # Popular balanced choice for RAG pipelines
    "o3-mini",            # Reasoning model; strong at multi-step SQL + RAG
    "o1-mini",            # Reasoning model; solid for analytical queries
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    # ── Anthropic (requires ANTHROPIC_API_KEY) ────────────────────────────────
    "claude-3-5-sonnet-20241022",   # Excellent reasoning + instruction following
    "claude-3-5-haiku-20241022",    # Fast & efficient; great RAG retrieval
    "claude-3-opus-20240229",       # Most capable Claude 3; strong for complex RAG
    # ── Google Gemini (requires GEMINI_API_KEY) ───────────────────────────────
    "gemini-2.0-flash",   # Fast; strong RAG with large context window
    "gemini-1.5-pro",     # Long context (1M tokens); deep document RAG
    "gemini-2.5-pro",     # Latest; top-tier reasoning + RAG accuracy
    # ── DeepSeek (requires DEEPSEEK_API_KEY) ─────────────────────────────────
    "deepseek-chat",      # DeepSeek-V3; competitive with GPT-4o for RAG
    "deepseek-reasoner",  # DeepSeek-R1; reasoning-focused, good for SQL+RAG
]

CATEGORIES = ["conversational", "sql", "performance"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    selected_model = st.selectbox(
        "Evaluation Model",
        AVAILABLE_MODELS,
        index=0,
        help="Model used as the CRM salesbot agent during evaluations",
    )

    # Default judge to gpt-4o-mini if available, otherwise first model
    _default_judge_idx = next(
        (i for i, m in enumerate(AVAILABLE_MODELS) if m == "gpt-4o-mini"), 0
    )
    selected_judge_model = st.selectbox(
        "Judge Model",
        AVAILABLE_MODELS,
        index=_default_judge_idx,
        help=(
            "Separate model used to score conversational responses. "
            "Can be a different provider than the evaluation model."
        ),
    )

    prompts_list = db.get_prompts()
    prompt_names = [p["name"] for p in prompts_list]
    default_idx = next(
        (i for i, p in enumerate(prompts_list) if p.get("is_default")), 0
    )
    selected_prompt_name = st.selectbox(
        "Active Prompt",
        prompt_names,
        index=default_idx,
        help="Prompt used for single-run evaluations",
    )
    selected_prompt = next(
        (p for p in prompts_list if p["name"] == selected_prompt_name), prompts_list[0]
    )

    st.divider()
    tc_count = db.count_test_cases()
    st.metric("Total Test Cases", tc_count)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_tests, tab_prompts, tab_run, tab_results, tab_compare = st.tabs(
    ["📋 Test Cases", "✏️ Prompts", "▶️ Run Evaluation", "📊 Results", "🔀 Compare"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEST CASES
# ══════════════════════════════════════════════════════════════════════════════
with tab_tests:
    st.header("Test Case Manager")

    sub_conv, sub_sql, sub_perf = st.tabs(
        ["Conversational", "SQL", "Performance"]
    )

    def _render_rubric(category: str):
        rubric = db.RUBRICS.get(category, [])
        if not rubric:
            return
        with st.expander("📐 Scoring Rubric"):
            if category == "conversational":
                st.dataframe(
                    pd.DataFrame(rubric),
                    column_config={
                        "dimension":   st.column_config.TextColumn("Dimension"),
                        "weight":      st.column_config.TextColumn("Weight"),
                        "description": st.column_config.TextColumn("Description"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.dataframe(
                    pd.DataFrame(rubric),
                    column_config={
                        "criterion":   st.column_config.TextColumn("Criterion"),
                        "description": st.column_config.TextColumn("Description"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

    def _render_test_tab(category: str, tab):
        with tab:
            _render_rubric(category)

            cases = db.get_test_cases(category)

            col_add, col_del = st.columns([1, 1])

            # ── Add form ───────────────────────────────────────────────────────
            with col_add:
                with st.expander("➕ Add New Test Case"):
                    with st.form(f"add_{category}", clear_on_submit=True):
                        q = st.text_area("Question *", key=f"q_{category}")
                        if category == "sql":
                            g = st.text_area("Golden SQL", key=f"g_{category}")
                            er = st.number_input(
                                "Expected Rows (optional, 0 = skip)",
                                min_value=0, key=f"er_{category}",
                            )
                        elif category == "performance":
                            er = st.number_input("Expected Rows (optional, 0 = skip)", min_value=0, key=f"er_{category}")
                            tt = st.number_input("Time Threshold (ms)", min_value=100, value=10000, key=f"tt_{category}")
                        submitted = st.form_submit_button("Add")
                        if submitted:
                            if not q.strip():
                                st.error("Question is required.")
                            else:
                                if category == "sql":
                                    db.add_test_case(
                                        category=category,
                                        question=q.strip(),
                                        golden_sql=g.strip() or None,
                                        expected_rows=int(er) if er else None,
                                    )
                                elif category == "performance":
                                    db.add_test_case(
                                        category=category,
                                        question=q.strip(),
                                        expected_rows=int(er) if er else None,
                                        time_threshold_ms=float(tt),
                                    )
                                else:
                                    db.add_test_case(category=category, question=q.strip())
                                st.success("Test case added.")
                                st.rerun()

            if not cases:
                st.info("No test cases yet. Add one using the form above.")
                return

            # ── Editable table ─────────────────────────────────────────────────
            df = pd.DataFrame(cases)
            display_cols = ["id", "question"]
            if category == "sql":
                display_cols += ["golden_sql", "expected_rows"]
            elif category == "performance":
                display_cols += ["expected_rows", "time_threshold_ms"]

            df_display = df[display_cols].copy()
            df_display.insert(0, "select", False)

            edited = st.data_editor(
                df_display,
                column_config={
                    "select":            st.column_config.CheckboxColumn("Del?", width="small"),
                    "id":                st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "question":          st.column_config.TextColumn("Question", width="large"),
                    "golden_sql":        st.column_config.TextColumn("Golden SQL", width="large"),
                    "expected_rows":     st.column_config.NumberColumn("Exp. Rows", help="Leave blank to skip row-count check"),
                    "time_threshold_ms": st.column_config.NumberColumn("Threshold (ms)"),
                },
                use_container_width=True,
                key=f"editor_{category}",
                num_rows="fixed",
            )

            # Save inline edits
            if st.button("💾 Save Edits", key=f"save_{category}"):
                for _, row in edited.iterrows():
                    tc_id = int(row["id"])
                    updates = {"question": row["question"]}
                    if category == "sql":
                        updates["golden_sql"] = row.get("golden_sql")
                        updates["expected_rows"] = (
                            int(row["expected_rows"]) if pd.notna(row.get("expected_rows")) else None
                        )
                    elif category == "performance":
                        updates["expected_rows"] = (
                            int(row["expected_rows"]) if pd.notna(row.get("expected_rows")) else None
                        )
                        updates["time_threshold_ms"] = float(row.get("time_threshold_ms", 10000))
                    db.update_test_case(tc_id, **updates)
                st.success("Changes saved.")
                st.rerun()

            # Delete selected rows
            with col_del:
                to_delete = edited[edited["select"] == True]["id"].tolist()
                if to_delete:
                    if st.button(f"🗑️ Delete {len(to_delete)} selected", key=f"del_{category}",
                                 type="primary"):
                        for tc_id in to_delete:
                            db.delete_test_case(int(tc_id))
                        st.success(f"Deleted {len(to_delete)} test case(s).")
                        st.rerun()

    _render_test_tab("conversational", sub_conv)
    _render_test_tab("sql", sub_sql)
    _render_test_tab("performance", sub_perf)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROMPTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_prompts:
    st.header("Prompt Manager")
    st.caption(
        "Create and edit named prompt variants. Each variant has a **System Prompt** "
        "(agent persona) and a **SQL Generation Prompt** (SQL instruction template). "
        "Use `{schema}`, `{context}`, and `{question}` as placeholders in the SQL prompt."
    )

    prompts = db.get_prompts()

    # ── New prompt form ────────────────────────────────────────────────────────
    with st.expander("➕ Create New Prompt Variant"):
        with st.form("new_prompt", clear_on_submit=True):
            p_name = st.text_input("Name *", placeholder="e.g. Prompt v2 — Concise")
            p_desc = st.text_input("Description", placeholder="Short description of this variant")
            p_sys = st.text_area(
                "System Prompt *",
                height=200,
                placeholder="You are a helpful sales assistant…",
            )
            p_sql = st.text_area(
                "SQL Generation Prompt *",
                height=200,
                placeholder="Use {schema}, {context}, {question} as placeholders",
            )
            if st.form_submit_button("Create Prompt"):
                if not p_name.strip() or not p_sys.strip() or not p_sql.strip():
                    st.error("Name, System Prompt, and SQL Prompt are required.")
                else:
                    db.add_prompt(p_name.strip(), p_desc.strip(), p_sys.strip(), p_sql.strip())
                    st.success(f"Prompt '{p_name}' created.")
                    st.rerun()

    st.divider()

    # ── List existing prompts ──────────────────────────────────────────────────
    for prompt in prompts:
        badge = " 🌟 Default" if prompt.get("is_default") else ""
        with st.expander(f"**{prompt['name']}**{badge}  —  {prompt.get('description', '')}"):
            with st.form(f"edit_prompt_{prompt['id']}"):
                e_name = st.text_input("Name", value=prompt["name"])
                e_desc = st.text_input("Description", value=prompt.get("description") or "")
                e_sys = st.text_area("System Prompt", value=prompt["system_prompt"], height=250)
                e_sql = st.text_area("SQL Generation Prompt", value=prompt["sql_prompt"], height=250)

                c1, c2, c3 = st.columns(3)
                save_btn = c1.form_submit_button("💾 Save")
                default_btn = c2.form_submit_button("🌟 Set Default")
                delete_btn = c3.form_submit_button("🗑️ Delete", type="primary")

                if save_btn:
                    db.update_prompt(prompt["id"], e_name, e_desc, e_sys, e_sql)
                    st.success("Prompt updated.")
                    st.rerun()

                if default_btn:
                    db.set_default_prompt(prompt["id"])
                    st.success(f"'{e_name}' is now the default prompt.")
                    st.rerun()

                if delete_btn:
                    try:
                        db.delete_prompt(prompt["id"])
                        st.success("Prompt deleted.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RUN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.header("Run Evaluation")

    all_prompts = db.get_prompts()

    col_cfg, col_run = st.columns([1, 2])

    with col_cfg:
        st.subheader("Configuration")

        run_name_input = st.text_input(
            "Run Name",
            value=f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )

        st.write("**Categories to evaluate:**")
        do_conv = st.checkbox("Conversational", value=True)
        do_sql = st.checkbox("SQL", value=True)
        do_perf = st.checkbox("Performance", value=False)

        st.write("**Prompt(s) to run:**")
        prompt_selections = {}
        for p in all_prompts:
            prompt_selections[p["id"]] = st.checkbox(
                p["name"],
                value=p.get("is_default", False),
                key=f"run_prompt_{p['id']}",
            )

        selected_prompt_ids = [pid for pid, checked in prompt_selections.items() if checked]

        st.write("**Models:**")
        run_model = st.selectbox(
            "Evaluation model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(selected_model),
            key="run_model_select",
        )
        run_judge_model = st.selectbox(
            "Judge model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(selected_judge_model),
            key="run_judge_model_select",
            help="Model used to score conversational responses (can differ from evaluation model)",
        )

    with col_run:
        st.subheader("Progress")

        if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
            if not any([do_conv, do_sql, do_perf]):
                st.error("Select at least one category to evaluate.")
            elif not selected_prompt_ids:
                st.error("Select at least one prompt.")
            else:
                # Build list of (category, test_cases) pairs to run
                run_queue = []
                if do_conv:
                    run_queue += [("conversational", tc) for tc in db.get_test_cases("conversational")]
                if do_sql:
                    run_queue += [("sql", tc) for tc in db.get_test_cases("sql")]
                if do_perf:
                    run_queue += [("performance", tc) for tc in db.get_test_cases("performance")]

                if not run_queue:
                    st.warning("No test cases found. Add test cases in the Test Cases tab.")
                else:
                    total_work = len(run_queue) * len(selected_prompt_ids)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_placeholder = st.empty()

                    done = 0
                    summary_rows = []

                    for prompt_id in selected_prompt_ids:
                        prompt_row = db.get_prompt(prompt_id)
                        prompt_dict = {
                            "system_prompt": prompt_row["system_prompt"],
                            "sql_prompt": prompt_row["sql_prompt"],
                        }
                        run_id = db.create_run(
                            run_name_input, run_model, prompt_id, prompt_row["name"]
                        )
                        passed_count = 0
                        failed_count = 0

                        for category, tc in run_queue:
                            status_text.markdown(
                                f"**Prompt:** {prompt_row['name']} | "
                                f"**Category:** {category} | "
                                f"**Q:** {tc['question'][:60]}…"
                            )

                            try:
                                if category == "conversational":
                                    result = evaluator.run_conversational_test(
                                        tc["question"], run_model, prompt_dict,
                                        judge_model=run_judge_model,
                                    )
                                elif category == "sql":
                                    result = evaluator.run_sql_test(
                                        tc["question"],
                                        tc.get("golden_sql") or "",
                                        run_model,
                                        prompt_dict,
                                        expected_rows=tc.get("expected_rows"),
                                    )
                                else:  # performance
                                    result = evaluator.run_performance_test(
                                        tc["question"],
                                        tc.get("expected_rows"),
                                        tc.get("time_threshold_ms") or 10000.0,
                                        run_model, prompt_dict,
                                    )
                            except Exception as exc:
                                result = {
                                    "passed": False,
                                    "llm_response": f"Error: {exc}",
                                    "score": 0.0,
                                    "error_message": str(exc),
                                }

                            db.save_result(run_id, tc.get("id"), category, tc["question"], result)

                            if result.get("passed"):
                                passed_count += 1
                            else:
                                failed_count += 1

                            summary_rows.append({
                                "Prompt": prompt_row["name"],
                                "Category": category,
                                "Question": tc["question"][:70],
                                "Pass": "✅" if result.get("passed") else "❌",
                                "Score": result.get("score"),
                            })

                            done += 1
                            progress_bar.progress(done / total_work)

                        db.finalize_run(run_id, len(run_queue), passed_count, failed_count)

                    status_text.markdown("✅ **Evaluation complete!**")
                    results_placeholder.dataframe(
                        pd.DataFrame(summary_rows), use_container_width=True
                    )
                    st.success(
                        f"Finished {total_work} tests across {len(selected_prompt_ids)} prompt(s). "
                        "View detailed results in the Results tab."
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.header("Results Viewer")

    runs = db.get_runs()
    if not runs:
        st.info("No evaluation runs yet. Go to the Run Evaluation tab to get started.")
    else:
        run_options = {
            f"[{r['id']}] {r['run_name']} — {r['model']} / {r['prompt_name']} "
            f"({r['passed']}/{r['total_cases']} passed)": r["id"]
            for r in runs
        }
        selected_run_label = st.selectbox("Select Run", list(run_options.keys()))
        selected_run_id = run_options[selected_run_label]
        run_meta = next(r for r in runs if r["id"] == selected_run_id)

        # Summary metrics
        total = run_meta["total_cases"] or 0
        passed = run_meta["passed"] or 0
        failed = run_meta["failed"] or 0
        pass_rate = (passed / total * 100) if total > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tests", total)
        m2.metric("Passed", passed)
        m3.metric("Failed", failed)
        m4.metric("Pass Rate", f"{pass_rate:.1f}%")

        st.divider()

        results = db.get_results(selected_run_id)
        if not results:
            st.info("No results for this run.")
        else:
            # Category filter
            categories_present = list({r["category"] for r in results})
            cat_filter = st.multiselect(
                "Filter by category", categories_present, default=categories_present
            )
            pass_filter = st.radio(
                "Show", ["All", "Passed only", "Failed only"], horizontal=True
            )

            filtered = [
                r for r in results
                if r["category"] in cat_filter
                and (
                    pass_filter == "All"
                    or (pass_filter == "Passed only" and r.get("passed"))
                    or (pass_filter == "Failed only" and not r.get("passed"))
                )
            ]

            for r in filtered:
                icon = "✅" if r.get("passed") else "❌"
                score_str = f"  Score: {r['score']:.2f}" if r.get("score") is not None else ""
                label = f"{icon} [{r['category'].upper()}]{score_str}  —  {str(r['question'])[:80]}"
                with st.expander(label):
                    col_l, col_r = st.columns(2)

                    with col_l:
                        st.markdown("**Question**")
                        st.write(r["question"])

                        st.markdown("**LLM Response**")
                        st.write(r.get("llm_response") or "_No response_")

                        if r.get("error_message"):
                            st.error(f"Error: {r['error_message']}")

                    with col_r:
                        if r.get("generated_sql"):
                            st.markdown("**Generated SQL**")
                            st.code(r["generated_sql"], language="sql")

                        # Conversational sub-scores
                        if r["category"] == "conversational" and r.get("relevance") is not None:
                            st.markdown("**Dimension Scores (1–5)**")
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Relevance (25%)", r["relevance"])
                            sc2.metric("Accuracy (30%)", r["accuracy"])
                            sc3.metric("Completeness (20%)", r["completeness"])
                            sc4, sc5 = st.columns(2)
                            sc4.metric("Actionability (10%)", r["actionability"])
                            sc5.metric("Safety (15%)", r["safety"])

                        # Performance metrics
                        if r.get("total_time_ms") is not None:
                            st.markdown("**Performance Metrics**")
                            pm1, pm2, pm3 = st.columns(3)
                            pm1.metric("LLM Latency", f"{r.get('llm_latency_ms', 0):.0f} ms")
                            pm2.metric("Total Time", f"{r.get('total_time_ms', 0):.0f} ms")
                            pm3.metric("Rows Returned", r.get("rows_returned") or "—")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("Compare Runs")
    st.caption("Select 2–4 evaluation runs to compare pass rates and per-question scores side-by-side.")

    runs = db.get_runs()
    if len(runs) < 2:
        st.info("You need at least 2 completed evaluation runs to compare. Run the evaluation first.")
    else:
        run_labels = {
            f"[{r['id']}] {r['run_name']} — {r['model']} / {r['prompt_name']}": r["id"]
            for r in runs
        }
        selected_labels = st.multiselect(
            "Select runs to compare (2–4)",
            list(run_labels.keys()),
            default=list(run_labels.keys())[:min(2, len(run_labels))],
        )

        if len(selected_labels) < 2:
            st.warning("Select at least 2 runs.")
        elif len(selected_labels) > 4:
            st.warning("Select at most 4 runs.")
        else:
            selected_ids = [run_labels[lbl] for lbl in selected_labels]
            selected_runs = [r for r in runs if r["id"] in selected_ids]

            # ── Summary metrics side-by-side ───────────────────────────────────
            st.subheader("Pass Rate Summary")
            metric_cols = st.columns(len(selected_runs))
            for col, run in zip(metric_cols, selected_runs):
                total = run["total_cases"] or 0
                passed = run["passed"] or 0
                rate = (passed / total * 100) if total > 0 else 0
                col.metric(
                    f"{run['prompt_name']}\n({run['model']})",
                    f"{rate:.1f}%",
                    f"{passed}/{total} passed",
                )

            st.divider()
            st.subheader("Per-Question Comparison")

            # Build a pivot: question → {run_id: pass/score}
            all_results = db.get_results_for_runs(selected_ids)
            if not all_results:
                st.info("No results found for selected runs.")
            else:
                run_id_to_label = {
                    r["id"]: f"{r['prompt_name']} / {r['model']}"
                    for r in selected_runs
                }

                # Group by (category, question)
                from collections import defaultdict
                pivot: dict = defaultdict(dict)
                for res in all_results:
                    key = (res["category"], res["question"])
                    run_label = run_id_to_label.get(res["run_id"], str(res["run_id"]))
                    passed_icon = "✅" if res.get("passed") else "❌"
                    score_val = res.get("score")
                    cell = passed_icon
                    if score_val is not None:
                        cell += f" ({score_val:.2f})"
                    pivot[key][run_label] = cell

                # Category filter
                cats_in_compare = list({k[0] for k in pivot})
                cat_filt = st.multiselect(
                    "Filter by category", cats_in_compare, default=cats_in_compare,
                    key="compare_cat_filter"
                )

                run_labels_ordered = [run_id_to_label[rid] for rid in selected_ids]
                rows = []
                for (cat, question), run_cells in pivot.items():
                    if cat not in cat_filt:
                        continue
                    row = {"Category": cat, "Question": question[:80]}
                    for rl in run_labels_ordered:
                        row[rl] = run_cells.get(rl, "—")
                    # Highlight rows where runs disagree
                    pass_vals = [v.startswith("✅") for v in run_cells.values()]
                    row["Differs?"] = "⚠️" if len(set(pass_vals)) > 1 else ""
                    rows.append(row)

                if rows:
                    df_cmp = pd.DataFrame(rows)
                    # Show differing rows first
                    df_cmp = df_cmp.sort_values("Differs?", ascending=False)
                    st.dataframe(df_cmp, use_container_width=True, hide_index=True)
                    differ_count = df_cmp["Differs?"].str.contains("⚠️").sum()
                    st.caption(
                        f"{differ_count} question(s) have different pass/fail outcomes across runs."
                    )
                else:
                    st.info("No matching results for the selected filters.")
