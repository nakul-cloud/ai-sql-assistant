# ✍️ Prompting Manual & Engineering Guide

This document acts as a comprehensive registry and engineering guide for all **prompt templates** used within the **AI SQL Analytics Assistant**. It details the structural prompting paradigms, few-shot examples, safety constraints, and conversational alignment rules that power our system.

---

## 🗺️ Prompting Architecture Overview

The system divides prompting into four logical stages to separate concerns and optimize model focus:

```
[User Input]
     │
     ▼
[Contextualization Prompt] ──► Rephrases using mem0 Cloud facts
     │
     ▼
[Intent Classifier Prompt] ──► Routes to 1 of 9 intents (e.g. SQL, CHAT)
     │
     ▼
[SQL Generation Prompt]    ──► Transforms schema metadata into T-SQL SELECT
     │
     ▼
[NL Response Prompt]       ──► Summarizes data profiles conversationally
```

---

## 🔍 Detailed Prompt Templates

### 1. SQL Generation Prompt (`SQL_PROMPT`)
* **File Reference**: [llm/query_ai.py](file:///d:/ai-sql-assistant/llm/query_ai.py)
* **Goal**: Generates highly optimized, database-safe, syntactically correct Microsoft SQL Server (T-SQL) queries.

#### Prompt Structure & Rules:
* **Senior Developer Persona**: Forces ChatGroq to assume the perspective of a senior T-SQL engineer to generate professional, performant queries.
* **Scope Boundary Guard**: Limits the model entirely to the provided schema context. If questions fall outside of the active database schema, the model is strictly instructed to return `OUT_OF_SCOPE` or `CANNOT_GENERATE` instead of hallucinating.
* **T-SQL Syntax Controls**:
  * Enforces `TOP [N]` instead of `LIMIT [N]`.
  * Enforces correct placing of keywords (e.g. `SELECT DISTINCT TOP [N] ...` instead of `SELECT TOP [N] DISTINCT ...`).
  * Enforces explicitly named columns instead of `SELECT *`.
  * Enforces NULL-safe comparisons (`IS NULL` or `IS NOT NULL` instead of `= NULL`).
* **Few-Shot Aggregation Demonstrations**:
  * Guides the model to output standard table-wide aggregations (e.g., `COUNT(*)`, `AVG()`, `MIN()`, `MAX()`) without a `TOP` clause when users ask overview or statistics questions (e.g., *"what does this dataset tell us"*). This prevents the AI from generating a record-level select which would crop the dataset to a small sample.
* **Comparative Context Pattern**:
  * Directs the LLM to write CTEs and window functions when queries filter for specific entities. This provides baseline average and maximum metrics alongside the target results for conversational comparisons.

---

### 2. Conversational Response Prompt (`NL_PROMPT`)
* **File Reference**: [llm/response_generator.py](file:///d:/ai-sql-assistant/llm/response_generator.py)
* **Goal**: Interprets database results and summarizes them for business users.

#### Core Guardrails & Framing Rules:
* **Strict Evidence Grounding**: Restricts the LLM from speculating about business impact, causation, or future outcomes. It must only state values present in the query results.
* **No Currency Assumptions**: Blocks the model from injecting currency symbols ($, €, ₹) unless the symbol or currency name is explicitly present in the data rows or column headers. Rounds all decimals to at most 2 decimal places.
* **Speculation Blocklist**: Enforces a strict ban on speculative vocabulary, blocking words like *"suggests"*, *"indicates"*, *"could imply"*, and *"likely due to"*.
* **Sample vs. Full-Dataset Claims**:
  * If the result was truncated (`is_truncated=YES`), the prompt instructs the model to describe only the rows shown and warns against applying averages or aggregates computed from this subset to the entire table.
  * If the result represents a full calculation (`is_truncated=NO`), the model describes the aggregates as definitive facts about the dataset.
* **Framing Isolation & Verbatim Blocking**:
  * Instructs the LLM to ignore past caveats or framing from the conversation history (e.g., if a previous query was truncated, that caveat must not bleed into a subsequent full query response).
  * Strictly blocks copying formatting artifacts, backticks, or markdown code-spans from past history logs, ensuring fresh prose.

---

### 3. Intent Routing Prompt (`INTENT_PROMPT`)
* **File Reference**: [retrieval/query_router.py](file:///d:/ai-sql-assistant/retrieval/query_router.py)
* **Goal**: Classifies user queries into nine operational buckets.

#### Intent Mapping:
* `CHAT`: Conversation, greetings, thanks.
* `SQL_QUERY`: Requests for lists, summaries, counts, and statistical data.
* `SCHEMA_INFO`: Queries asking what tables or columns exist in the database.
* `DESCRIBE`: Asks to explain the database tables or overview schemas.
* `DATA_PREVIEW`: Direct request to see rows (triggers a direct fast SELECT without LLM generation).
* `SCHEMA_EXPLANATION`: Asks for definitions of specific column names or abbreviations.
* `CONVERSATION_SUMMARY`: Asks to summarize topics previously discussed.
* `TEMPORAL`: Requests about today's date/time (triggers a scope-boundary warning).
* `GENERAL_KNOWLEDGE`: Asks about real-world facts (triggers an out-of-scope warning).

---

### 4. Contextualization & Rephrasing Prompt (`CONTEXTUALIZE_PROMPT`)
* **File Reference**: [memory/mem0_manager.py](file:///d:/ai-sql-assistant/memory/mem0_manager.py)
* **Goal**: Resolves conversational pronouns and makes questions standalone.

#### Rephrasing Logic:
* Takes the **Conversation History**, **Retrieved mem0 Semantic Facts**, and the **Latest Follow-up Question**.
* Rephrases the input to resolve ambiguous references.
* *Example*:
  - *User*: "What about female students?"
  - *Facts*: "User is querying the student placements table."
  - *Output*: "Show placement information for female students."

---

## 💡 Engineering Best Practices & Rationale

> [!IMPORTANT]
> **Why Temperature Matters**
> We configure model temperatures strictly per chain:
> * **SQL Generation and Classification**: `temperature=0.0` to maximize precision, eliminate creative hallucinations, and enforce strict syntax rules.
> * **NL Response Synthesis**: `temperature=0.3` to allow fluid, professional business summaries while remaining heavily grounded.

> [!TIP]
> **Few-Shot Prompting Strategy**
> Rather than explaining complex formatting rules in abstract prose, we embed concrete input-output examples in our prompt templates. This dramatically improves formatting compliance and T-SQL parsing.

> [!WARNING]
> **Strict Guardrails Overrule Generosity**
> If there is any ambiguity about whether the database schema can answer a user query, the generation rules prioritize safety. Returning `OUT_OF_SCOPE` blocks the model from guessing, protecting downstream dashboards from incorrect or hallucinated SQL execution.
