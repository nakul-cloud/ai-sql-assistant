# Analysis Layer

This directory contains utility modules for analyzing database schemas and formatting them into prompts for the Large Language Model.

---

## File Registry

### 1. `schema_context.py`
Builds schema context blocks for prompt generation:
- **`generate_schema_context()`**: Generates structured, readable schema text for selected tables.
- **Context Synthesis**: For each table, it compiles:
  - Table name and row counts.
  - Primary key columns.
  - Detailed column lists (including data types, nullability, and primary key designations).
  - Sample value lists to help the model write accurate queries.
- **Prompt Grounding**: Formats the schema context to guide the LLM's query generation, preventing table/column hallucinations.

### 2. `result_enricher.py`
Provides the semantic enrichment layer for SQL execution results:
- **`enrich_sql_result()`**: Analyzes raw SQL execution records using Pandas.
- **Statistical Summaries**: Generates lightweight profiles containing sums, means, minimums, maximums for numbers, and unique/top values for categorical data.
- **Payload Optimization**: Shrinks massive query results (e.g., 1000+ rows) down to a tiny, token-efficient metadata summary and a 5-row sample. This prevents LLM context overload and ensures lightning-fast natural language responses.

---

## Generated Prompt Context Format

The module generates schema details using this format:

```text
Table: dbo.csv_employees
Row Count: 5
Primary Keys: employee_id
Columns:
  - employee_id (bigint, NOT NULL) [PK]
  - employee_name (varchar(MAX), NULL)
  - salary (bigint, NULL)
Sample Values:
  - employee_id: ['101', '102', '103']
  - employee_name: ['John Doe', 'Jane Doe', 'Sam Spade']
  - salary: ['95000', '105000', '72000']
```

---

## Verification

Test prompt generation from the command line:

```bash
.venv\Scripts\python.exe -m analysis.schema_context
```
