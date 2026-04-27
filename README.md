# LLM Evaluation Suite

A Streamlit app for evaluating RAG chatbot quality across **Conversational**, **SQL**, and **Performance** dimensions. Supports multi-model evaluation using OpenAI, Anthropic, Google Gemini, and DeepSeek — with a separate LLM-as-judge for scoring conversational responses.

## Features

- **Multi-model evaluation** — pick any model as the agent under test and a separate model as the judge
- **Three evaluation categories** — conversational (weighted rubric scoring), SQL (correctness + row count), performance (latency + row count)
- **Editable rubric** — adjust scoring dimension weights and descriptions from the UI
- **Prompt versioning** — create and compare multiple system/SQL prompt variants
- **Run comparison** — side-by-side pass rate and per-question comparison across runs
- **CSV export** — download results for any run or comparison

## Project Structure

```
eval_app/
├── app.py          # Streamlit UI (5 tabs: Test Cases, Prompts, Run, Results, Compare)
├── evaluator.py    # Core evaluation engine (Streamlit-agnostic; usable from CLI/notebooks)
├── llm_client.py   # Unified API client (auto-detects provider from model name)
├── judge.py        # LLM-as-judge for conversational scoring
├── eval_db.py      # DuckDB database layer (test cases, runs, results, rubric)
├── importer.py     # Excel seeder for test cases and prompts
├── requirements.txt
└── data/
    ├── Capstone_Final.xlsx           # Test cases seed data
    └── Capstone Prompts Final.xlsx   # Prompt variants seed data
Reference Files/
└── rag_salesbot-main/          # Required — eval app imports from here at runtime
    ├── app/                    # Python modules imported by evaluator.py
    │   ├── database/           # DB connection and schema helpers
    │   └── agent/              # Tool handlers
    └── db/
        └── sales.duckdb        # CRM database used during evaluations
```

## Setup

**1. Install dependencies**

```bash
pip install -r eval_app/requirements.txt
```

**2. Set API keys**

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=...       # optional — only needed for Gemini models
export DEEPSEEK_API_KEY=...     # optional — only needed for DeepSeek models
```

> **No quotes around the key value.** Copying a quoted command from a rendered document (GitHub, PDF, Word) can silently replace straight quotes with curly/smart quotes, which get included as part of the key and cause an `'ascii' codec` error. If you see that error, run `echo $OPENAI_API_KEY` — if the value starts with `'`, re-export without quotes.

**3. Run the app**

```bash
cd eval_app
streamlit run app.py
```

Opens at `http://localhost:8501`. On first launch, the app creates `eval.duckdb` and seeds it with default test cases, prompts, and rubric weights from the included Excel files (if present).

> **Run locally, not on Streamlit Community Cloud.** Evaluation runs are long-running synchronous loops (with configurable delays between API calls). Streamlit Cloud's connection timeouts will kill runs mid-flight, and its ephemeral filesystem means your DuckDB data won't persist across restarts.

## Data Persistence

All data is stored in `eval_app/eval.duckdb`, a local DuckDB file created on first launch.

**What persists across restarts:** everything — test cases, prompts, rubric edits, evaluation runs, and results.

**Initial seeding:** on the very first launch (when the database is empty), the app seeds test cases and prompts from the Excel files in `eval_app/data/`:

- `Capstone_Final.xlsx` — seeds the Conversational, SQL, and Performance test case tabs
- `Capstone Prompts Final.xlsx` — seeds the Prompts tab with the initial prompt variants

After that first launch, seeding is skipped. Any test cases or prompts you add, edit, or delete through the UI are saved permanently to `eval.duckdb` and will still be there after a restart.

**To reset to the original seed data:** delete `eval_app/eval.duckdb` and restart the app. The database will be recreated and re-seeded from the Excel files.

## Usage

| Tab | What it does |
|-----|-------------|
| **Test Cases** | Add, edit, and delete test cases per category. Edit scoring rubric weights and descriptions. |
| **Prompts** | Create and manage system prompt + SQL prompt variants. Set a default. |
| **Run Evaluation** | Select model, judge model, categories, and prompts, then run a full evaluation batch. |
| **Results** | View pass/fail metrics, per-test details, judge reasoning, and token usage. Export to CSV. |
| **Compare** | Side-by-side comparison of two or more runs, with disagreement highlighting. |

## Evaluation Categories

**Conversational** — The LLM judge scores each response on five dimensions (1–5 scale), weighted and combined into a single score. Pass threshold: weighted score ≥ 3.0.

| Dimension | Default Weight |
|-----------|---------------|
| Relevance | 25% |
| Accuracy | 30% |
| Completeness | 20% |
| Actionability | 10% |
| Safety | 15% |

Weights are editable from the Test Cases tab and auto-normalized.

**SQL** — Pass/fail based on: valid SQL generated, successful execution, result accuracy, and optional row count match against an expected value.

**Performance** — Pass/fail based on: successful execution, total latency ≤ threshold (ms), and optional row count match.

## Supported Models

The app includes presets for models from four providers. The correct API key must be set for each provider used.

| Provider | Example Models |
|----------|---------------|
| OpenAI | `gpt-4o`, `gpt-4.1`, `gpt-4o-mini`, `o3-mini` |
| Anthropic | `claude-sonnet-4-6`, `claude-opus-4-7`, `claude-haiku-4-5-20251001` |
| Google | `gemini-2.5-pro`, `gemini-2.0-flash`, `gemini-1.5-pro` |
| DeepSeek | `deepseek-chat`, `deepseek-reasoner` |

The evaluation model and judge model can be from different providers.
