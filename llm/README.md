# Large Language Model (LLM) Layer

This directory handles prompt contexts, runs LangChain LCEL chains, and validates generated SQL statements to prevent security issues.

---

## Processing Pipeline

```mermaid
flowchart LR
    SchemaCtx[Schema Context] & UserQ[User Query] --> SQLGen[query_ai.py]
    SQLGen -->|1. Generate SQL| LCEL1[LangChain SQL Chain]
    LCEL1 -->|2. Clean SQL| Cleaner[SQL Cleaner]
    Cleaner -->|3. Regex Guardrails| Validator[SQL Validator]
    Validator -->|Output Safe SQL| Exec[Execution Output]
    
    Exec --> Enricher[result_enricher.py]
    Enricher & UserQ & SQLGen --> NLGen[response_generator.py]
    NLGen -->|4. Summarize Enriched Profile| LCEL2[LangChain NL Chain]
    LCEL2 -->|Conversational Summary| FinalAns[Plain English Answer]
```

---

## File Registry

### 0. `llm_client.py`
Provides the shared, cached `get_llm()` model provider using LangChain's `ChatGroq` wrapper. Uses a lazy-initialized singleton for raw completions (`generate_text`) to preserve offline indexing functionality. It supports fallback chains: Primary is Groq, falling back to OpenAI (GPT-4o-mini), and then Gemini (1.5 Flash).

### 1. `query_transformer.py`
Implements an **Advanced Query Transformation** layer that optimizes natural language inputs before they hit table retrieval or SQL generation. It applies three sequential strategies:
- **Query Decomposition**: Splits complex compound questions (e.g. joined by "and") into independent standalone sub-queries using LLM semantic parsing.
- **Query Rewriting (Schema Alignment)**: Maps colloquial terms to actual database schema vocabulary (e.g., matching synonyms) by examining active schema terms.
- **Step-Back Prompting**: Automatically generates a broader comparative search query for narrow, single-entity filters (e.g., "what about Healthcare?" -> "what is the AI adoption rate across all industries?") to provide background baseline metrics.

### 2. `query_ai.py`
Generates T-SQL queries from user questions using LangChain's ChatPromptTemplate and LCEL:
- **Prompt Isolation**: Directs the LLM using SQL Server T-SQL syntax guidelines (e.g., using `TOP` instead of `LIMIT`, referencing specific schema columns, wrapping spaces/reserved words in brackets).
- **Aggregate Rules**: Implements strict rules to generate table-wide aggregates without `TOP` clauses for overview/summary questions (e.g. "what does this dataset tell us?").
- **Output Sanitization**: Removes markdown formatting block tags (e.g. ` ```sql `) to produce clean SQL statements ready for execution.
- **SQL Validator (Security Guardrail)**: Checks generated SQL query blocks against security rules before running them on SQL Server:
  - **SELECT-Only Enforcement**: Ensures queries start with `SELECT` or `WITH`. Rejects updates, inserts, and deletions.
  - **Single Statement Enforcement**: Rejects queries with semicolons `;` that attempt to run multiple statements.
  - **Keyword Blocklist**: Filters out commands like `DROP`, `TRUNCATE`, `ALTER`, `EXEC`, and `CREATE`.
  - **Procedure Blocklist**: Blocks critical system stored procedures (e.g., `xp_cmdshell`, `sp_executesql`, `openrowset`).

### 3. `response_generator.py`
Converts database execution outcomes into clear, conversational summaries using LangChain:
- **Analyst Persona**: Guides the model to output summaries focused on business metrics, avoiding database terminology and table names.
- **Context Injection**: Ingests an enriched statistical profile (sums, means, unique counts) instead of raw datasets. This prevents token bloat and speeds up natural language generation.
- **Grounded Interpretation**: Bases response strictly on evidence. Blocks speculative storytelling, causation assumptions, or business impact projections.
- **No Currency Assumptions**: Prevents the model from prepending any currency symbols (like `$`, `€`, `₹`) unless the schema explicitly contains currency units.
- **Preview vs. Full-Dataset Claims**: Uses query metadata (`is_truncated`, `is_count_query`, `table_total_rows`) to accurately describe previews versus scalar aggregates. It ensures preview findings are never stated as full-dataset facts.
- **Framing Isolation & Verbatim Controls**: Isolates prompt history to prevent carrying over past sample-size caveats and forbids verbatim text reuse from history to maintain freshly composed prose.

### 4. `describe_generator.py`
Handles queries requesting details about specific datasets or database-wide summaries (e.g., "explain what this table is about"):
- **Dataset Description**: Translates table columns, metadata, and semantic cache descriptions into plain business language (e.g., "ai_investment_usd" -> "AI spending") without running database queries.
- **Database Overview**: Generates a high-level summary of all tables and approximate row counts.

### 5. `langchain_agent.py`
Executes an autonomous SQL agent fallback using LangChain's `create_sql_agent` with native tool-calling (`agent_type="tool-calling"`):
- **Self-Healing**: Triggered automatically if standard T-SQL execution fails. It connects to the database engine, inspects metadata, corrects syntax, and executes queries autonomously to recover from errors.


---

## Security Validation Policy

The SQL Validator checks generated SQL against these policies to keep database operations safe:

| Action | Allowed | Rationale |
|:---|:---:|:---|
| `SELECT ...` | ✅ Yes | Safe read-only data query. |
| `SELECT ...; DROP TABLE ...` | ❌ No | Blocks multi-statement injection. |
| `UPDATE ...` / `DELETE ...` | ❌ No | Prevents data modifications. |
| `EXEC xp_cmdshell ...` | ❌ No | Prevents remote server command execution. |
| `SELECT * FROM openrowset(...)` | ❌ No | Blocks unauthorized file reads or connections. |

---

## Verification

Test the components from the project root:

```bash
# Test LangChain client connection
.venv\Scripts\python.exe -m llm.llm_client

# Test SQL query generation and validation
.venv\Scripts\python.exe -m llm.query_ai

# Test conversational summary generation
.venv\Scripts\python.exe -m llm.response_generator

# Test autonomous tool-calling SQL agent
.venv\Scripts\python.exe -m llm.langchain_agent
```

---

## 🔄 Advanced Query Transformations

Before feeding the user's natural language question into the SQL generation chain, the system applies three query transformation patterns defined in `llm/query_transformer.py`:

### 1. Query Decomposition (Splitting Compound Intent)
* **Goal**: Handle complex queries that ask for multiple different metrics at once.
* **Mechanism**: A fast pre-check heuristic (`_has_compound_signals`) scans the query for conjunctions (`and`, `but`, `as well as`). If found, a ChatGroq-backed LCEL chain splits the query into standalone sub-questions.
* **Example**: 
  - *Input*: "Show me total sales for 2026, and who is the manager of HR department?"
  - *Decomposed Outputs*: 
    1. "What are the total sales for 2026?"
    2. "Who is the manager of the HR department?"
  - *Result*: The orchestrator runs both sub-questions through retrieval and SQL execution individually, then merges the final statistical insights in the response.

### 2. Schema Alignment & Synonym Rewriting
* **Goal**: Correct mapping between colloquial business terms and real database schema columns.
* **Mechanism**: If `_needs_rewrite` detects synonyms or slang terms, the rewriter maps these entities to the exact schema attributes using synonyms configured in metadata.
* **Example**:
  - *Input*: "list all our workers in California"
  - *Rewritten Output*: "List all rows in dbo.csv_employees where state is CA" (aligning "workers" to `dbo.csv_employees` and "California" to `state`).

### 3. Step-Back Prompting (Abstraction)
* **Goal**: Retrieve high-level baseline context to prevent data isolation in narrow filter queries.
* **Mechanism**: If `_might_need_stepback` detects a query focusing on a single entity or category (e.g. Sales, Technology, Male), it generates a broader step-back question to extract total/average benchmarks.
* **Example**:
  - *Input*: "What is the average salary of the Sales department?"
  - *Step-Back Query*: "What is the average salary across all departments?"
  - *Result*: The SQL generator writes queries to fetch both metrics, allowing the response generator to state: *"The Sales department average salary is 95,000, which is higher than the overall company average of 82,000."*

---

## ✍️ Types of Prompting in This Project

We utilize specialized prompting paradigms across the LLM chains to guarantee structured outputs and safe database interactions:

### 1. Persona-Driven Role Prompting
* **Usage**: In `query_ai.py` (`SQL_PROMPT`) and `response_generator.py` (`NL_PROMPT`).
* **Concept**: Tells the LLM exactly what role it plays, which alters its tone and vocabulary constraints.
* **Instruction**: 
  - *"You are a senior Microsoft SQL Server (T-SQL) engineer working on a business intelligence system."*
  - *"You are a production-grade AI Analytics Copilot... explain findings conversationally."*

### 2. Grounded Few-Shot Prompting
* **Usage**: Guides SQL generation for edge cases (e.g., aggregations vs. previews) and guides NL responses on formatting.
* **Concept**: Provides concrete examples of inputs and expected outputs to set the standard for parsing.
* **Instruction**: 
  - In SQL generation, examples demonstrate that overview/summary queries (e.g., *"what does this dataset tell us"*) must generate standard aggregations (e.g., `SELECT COUNT(*), AVG(x)`) *without* using a `TOP` clause.
  - In NL response generation, examples show how to formulate comparative answers utilizing natural pronouns.

### 3. Negative Constraints (Strict Guardrails & Refusals)
* **Usage**: Enforces security, privacy, and accuracy.
* **Concept**: Specifies exactly what the LLM is prohibited from generating.
* **Instruction**:
  - In SQL generation: *"Only SELECT statements. Never generate: DROP, DELETE, TRUNCATE, ALTER..."* and *"If a column does not exist, do not invent it."*
  - In NL response: *"Strictly avoid phrases like: 'this suggests...', 'this may indicate...'"* and *"Do NOT assume or inject any currency symbols."*

### 4. Comparative & Context-Aware Prompting
* **Usage**: Directs the query generator to construct advanced queries with window functions and CTEs.
* **Concept**: Ensures filtered queries gather relative statistics for context.
* **Instruction**: *"When the question filters on a specific category... do NOT just retrieve that single row. Write a query that retrieves the target metrics alongside overall table aggregates (average, maximum, rank)..."*

---

## 🔗 LangChain Expression Language (LCEL)

LCEL is a simple way to glue different AI components together using the pipe operator (`|`), just like unix terminal commands (`cat file.txt | grep "error"`).

Think of LCEL as a **factory assembly line** where data enters on one side, passes through different stations, and comes out fully processed on the other side.

### 🧱 The 3 Core Blocks of an LCEL Chain
In most chains, you connect three basic elements:
* **Prompt Template**: The instructions for the LLM (tells it what to do).
* **LLM (Model)**: The brain that processes the prompt (like Groq Llama 3).
* **Output Parser**: Cleans up the LLM's response (e.g., converts the raw AI response into a clean text string).

In Python, you chain them together using the pipe `|` symbol:
```python
chain = Prompt | LLM | OutputParser
```

### 🛠️ How We Use LCEL in This Project
We use LCEL to handle every text-processing pipeline in the codebase. Here are three real examples:

#### 1. Generating SQL Queries ([llm/query_ai.py](file:///d:/ai-sql-assistant/llm/query_ai.py))
This chain takes your question and table schemas and outputs clean SQL code:
```python
_sql_chain = SQL_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
```
* **How data flows**:
  1. `SQL_PROMPT` fills in your question and the schema context.
  2. The filled-in text is piped into the LLM (configured with `temperature=0.0` for precise, deterministic SQL).
  3. The raw response is piped into `StrOutputParser()` to strip away metadata and return a clean SQL string.

#### 2. Writing Conversational Summaries ([llm/response_generator.py](file:///d:/ai-sql-assistant/llm/response_generator.py))
This chain takes the database records and translates them into business answers:
```python
_nl_chain = NL_PROMPT | get_llm(temperature=0.3) | StrOutputParser()
```
* **How data flows**:
  1. `NL_PROMPT` receives the query, total row count, and statistical profile.
  2. It pipes it to the LLM (configured with `temperature=0.3` to allow for clean, natural-sounding sentences).
  3. `StrOutputParser` extracts the final natural language answer.

#### 3. Contextualizing Follow-up Questions ([workflow/process_query.py](file:///d:/ai-sql-assistant/workflow/process_query.py))
This chain reads chat history to rephrase ambiguous inputs:
```python
_contextualize_chain = CONTEXTUALIZE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()
```

### 🌟 Why Use LCEL? (The Benefits)
* **Automatic Streaming**: If you want the chat response to stream word-by-word on the screen, LCEL handles it out-of-the-box using `.stream()` instead of `.invoke()`.
* **Less Boilerplate**: You don't have to write code to extract text from deep nested JSON response objects (like `response.choices[0].message.content`). The output parser handles it automatically.
* **Unified Interface**: Every LCEL chain uses the same standard functions (`.invoke()`, `.stream()`, `.batch()`), making the codebase clean and easy to maintain.
