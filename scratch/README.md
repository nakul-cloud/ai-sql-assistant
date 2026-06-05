# Scratch Scripts

This directory houses developer scripts, benchmarks, and command-line test runners for validating the database assistant's components.

---

## File Registry

### 1. `run_rag_demo.py`
A command line runner for testing the database query pipeline:
- **Environment Isolation**: Configures Python's path (`sys.path`) dynamically so developers can run the tool from any folder without module path errors.
- **Direct CLI Execution**: Runs questions directly through the RAG pipeline, printing intent logs, retrieval rankings, database query times, and insights.
- **Usage**:
  ```bash
  .venv\Scripts\python.exe scratch/run_rag_demo.py "Show me all employees in the engineering department"
  ```

### 2. `test_warmup.py`
Benchmarks the startup time of the embedding model:
- Measures load and compilation times for the embedding models on CPU.
- Validates the performance optimizations implemented in `indexing/embedder.py`.
- **Usage**:
  ```bash
  .venv\Scripts\python.exe scratch/test_warmup.py
  ```
