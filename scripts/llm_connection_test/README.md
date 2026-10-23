# Model service connectivity

Run from the project root with its virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts/llm_connection_test/check_llm_connection.py
```

Optional arguments:

```text
--timeout SECONDS      Per-attempt timeout (default: 30)
--max-attempts COUNT   Attempts for retryable failures (default: 3)
--json                 Machine-readable output
```

Exit codes are `0` for success, `1` for a network/provider failure, `2` for
invalid configuration or arguments, and `3` for an unexpected model response.
The script never prints `RESEARCHOPS_MODEL_API_KEY`.
