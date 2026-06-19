---
name: semantic-enrichment
description: >
  Apply this skill whenever working on the response layer of the Enterprise AI SQL
  Analytics Assistant — specifically response_generator.py, query_executor.py result
  processing, or any step between SQL result rows and the final NL response shown to
  the user. Triggers on: "enrich results", "add context to response", "interpret data",
  "semantic enrichment", "pre-calculate", "add trend/status/risk to response",
  "response feels dumb", "agent-ready response", or any request to make the NL output
  more insightful. Do NOT apply to indexing, Qdrant, embeddings, or query routing —
  those are covered by the main SKILL.md.
---

# Semantic Enrichment Layer

## What This Solves

Your SQL query returns raw rows. Without enrichment, Gemini Flash sees:

```json
[
  {"ticket_id": 441, "status": "open", "created_date": "2024-10-01", "priority": 2},
  {"ticket_id": 442, "status": "open", "created_date": "2024-09-15", "priority": 1}
]
```

Gemini must then figure out: Is this overdue? Is 2 tickets a lot? What should the user do?
That's business logic — it doesn't belong inside the LLM's reasoning. It belongs in code.

With enrichment, Gemini sees:

```json
{
  "rows": [...],
  "summary": {
    "total": 2,
    "oldest_open_days": 81,
    "high_priority_count": 1,
    "status_breakdown": {"open": 2},
    "trend": "increasing",
    "alert": "1 ticket overdue by 81 days — past SLA threshold of 30 days"
  }
}
```

Now Gemini narrates a pre-understood situation instead of doing math.

---

## Where This Lives in the Pipeline

```
SQL Server result (raw rows)
        │
        ▼
 [NEW] result_enricher.py          ← insert here
        │  adds summary, trends, alerts, status fields
        ▼
 response_generator.py             ← Gemini Flash receives enriched context
        │
        ▼
 User sees NL answer + table
```

New file: `analysis/result_enricher.py`
Updated file: `workflow/process_query.py` — pass enriched context to `generate_natural_language_response()`

---

## The 80/20 Rule — What to Calculate vs. What to LLM

| Task | Method | Why |
|---|---|---|
| Count rows, totals, averages | Deterministic (Pandas) | Free, instant, reliable |
| Date math (overdue, age, SLA breach) | Deterministic | Exact, no hallucination risk |
| Status labels (overdue / on-track / critical) | Deterministic (rules) | Consistent, debuggable |
| Trend detection (increasing / stable / decreasing) | Deterministic | Pandas rolling window |
| Anomaly flags (outliers, SLA breaches) | Deterministic | Rule thresholds |
| Contextual narrative ("this is unusual because...") | LLM (Gemini Flash) | Only where rules can't capture nuance |

**Default: write a rule. Only escalate to LLM if a rule can't express it.**

---

## Core Enrichment Functions

File: `analysis/result_enricher.py`

### Function Signature

```python
def enrich_query_result(
    rows: list[dict],
    columns: list[str],
    user_query: str,
    sql_query: str
) -> dict:
    """
    Takes raw SQL result rows and returns an enriched context dict
    ready to pass to the NL response generator.

    Returns:
    {
        "rows": [...],          # original rows, unchanged
        "columns": [...],       # original columns, unchanged
        "summary": {...},       # computed stats and labels
        "alerts": [...],        # list of flagged conditions
        "interpretation": "..."  # optional: 1-line human-readable summary
    }
    """
```

### What `summary` Should Contain (computed with Pandas)

```python
summary = {
    "total_rows": len(df),
    "columns_returned": list(df.columns),

    # For numeric columns — auto-detect and summarize
    "numeric_summary": {
        col: {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": round(df[col].mean(), 2),
            "sum": df[col].sum()
        }
        for col in df.select_dtypes(include="number").columns
    },

    # For date columns — calculate age and overdue status
    "date_summary": {
        col: {
            "oldest": df[col].min(),
            "newest": df[col].max(),
            "oldest_age_days": (today - df[col].min()).days
        }
        for col in df.select_dtypes(include="datetime").columns
    },

    # For categorical/status columns — value counts
    "category_breakdown": {
        col: df[col].value_counts().to_dict()
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() <= 10  # only low-cardinality columns
    }
}
```

### Alert Rules (deterministic, business-aware)

Define rules as a list of `(condition_fn, alert_message)` pairs. Add rules specific to
your domain (tickets, orders, employees, invoices, etc.) as you discover patterns.

```python
ALERT_RULES = [
    # SLA / overdue checks — adjust thresholds to your business
    (
        lambda df: any_date_column_older_than(df, days=30),
        "⚠️ One or more records are older than 30 days — may be past SLA"
    ),
    (
        lambda df: "priority" in df.columns and (df["priority"] == 1).any(),
        "🔴 High-priority records present — review immediately"
    ),
    (
        lambda df: len(df) == 0,
        "ℹ️ No records found matching your criteria"
    ),
    (
        lambda df: len(df) > 100,
        f"📊 Large result set ({len(df)} rows) — showing summary; ask to filter further"
    ),
]

def compute_alerts(df: pd.DataFrame) -> list[str]:
    return [msg for condition, msg in ALERT_RULES if condition(df)]
```

### Trend Detection (when result has a date + numeric column)

```python
def detect_trend(df: pd.DataFrame, date_col: str, value_col: str) -> str:
    """
    Returns 'increasing', 'decreasing', 'stable', or 'insufficient_data'
    """
    if len(df) < 3:
        return "insufficient_data"

    df_sorted = df.sort_values(date_col)
    first_half = df_sorted[value_col].iloc[:len(df)//2].mean()
    second_half = df_sorted[value_col].iloc[len(df)//2:].mean()

    delta = (second_half - first_half) / (first_half + 1e-9)

    if delta > 0.1:
        return "increasing"
    elif delta < -0.1:
        return "decreasing"
    else:
        return "stable"
```

---

## Integration into `process_query.py`

Current flow (before enrichment):
```python
exec_result = execute_sql_query(sql_result["sql_query"])
nl_response = generate_natural_language_response(
    user_query,
    sql_result["sql_query"],
    exec_result["result"]          # ← raw rows go straight to Gemini
)
```

Updated flow (with enrichment):
```python
exec_result = execute_sql_query(sql_result["sql_query"])

# NEW: enrich before passing to NL generator
enriched = enrich_query_result(
    rows=exec_result["result"]["rows"],
    columns=exec_result["result"]["columns"],
    user_query=user_query,
    sql_query=sql_result["sql_query"]
)

nl_response = generate_natural_language_response(
    user_query,
    sql_result["sql_query"],
    enriched                       # ← enriched context goes to Gemini
)
```

---

## Updated NL Response Prompt

Update the prompt in `llm/response_generator.py` to explicitly use the enriched fields:

```python
prompt = f"""
You are a business data analyst assistant.
Answer the user's question using the pre-analyzed data context below.
The summary and alerts have already been calculated — use them directly.
Do not recalculate anything. Do not add speculation beyond the data.

User question: {user_query}
SQL used: {sql_query}

Pre-analyzed context:
- Total records: {enriched['summary']['total_rows']}
- Key statistics: {enriched['summary']['numeric_summary']}
- Alerts: {enriched['alerts']}
- Trend: {enriched['summary'].get('trend', 'N/A')}

Raw data sample (first 5 rows):
{enriched['rows'][:5]}

Write a clear, concise 2-4 sentence business answer.
Mention specific numbers. Reference alerts if present.
Do not mention SQL, databases, or technical terms.
"""
```

---

## Enrichment Decision Tree

When adding a new enrichment, use this decision tree:

```
Is the enrichment purely math or rule-based?
│
├── YES → Write it in result_enricher.py as a deterministic function
│         (date math, counts, thresholds, comparisons, trends)
│
└── NO → Does it require understanding patterns across multiple records
         or generating natural language explanations?
         │
         ├── YES → Pass the pre-calculated summary to Gemini Flash
         │         and let it narrate — don't write regex for this
         │
         └── NO → Can it be expressed as a threshold or category?
                  │
                  └── YES → Write a rule. Rules are free and reliable.
```

---

## Cost & Performance Impact

| Stage | Before enrichment | After enrichment |
|---|---|---|
| Gemini input tokens | ~500 (raw rows dumped as JSON) | ~200 (compact enriched summary) |
| Gemini reasoning required | High (must do math, date calc, comparisons) | Low (narrate pre-computed facts) |
| Response accuracy | Moderate (LLM math is unreliable) | High (math done in Python) |
| Latency added by enricher | — | ~10–30ms (Pandas, in-memory) |
| LLM cost | Baseline | ~40–60% reduction in tokens |

---

## Build Order for This Component

Build after `workflow/process_query.py` is wired up end-to-end.

1. Create `analysis/result_enricher.py` with `enrich_query_result()`
2. Add basic summary (row count, numeric stats, category breakdown)
3. Add alert rules relevant to your domain
4. Update `process_query.py` to call enricher before NL generator
5. Update prompt in `response_generator.py` to reference enriched fields
6. Test with 5 diverse queries — verify alerts fire correctly and NL response improves
7. Add trend detection if you have time-series data

Test it standalone:
```bash
python -m analysis.result_enricher
```
with a hardcoded sample result before integrating into the pipeline.

---

## Key Principle (from the article, applied here)

> "If your APIs only expose data, you're feeding intelligence with noise."

In this project: if `response_generator.py` receives raw rows, you're asking Gemini to
be a calculator. Let Python be the calculator. Let Gemini be the narrator.

**Deterministic enrichment = fast, free, reliable.**
**LLM narration of pre-computed facts = accurate, concise, trustworthy.**