# ResearchOps LLM connection test

Run from any directory with the project's virtual environment:

```powershell
D:\my_coach\new_agent\researchops_mvp\.venv\Scripts\python.exe `
  D:\my_coach\new_agent\researchops_mvp\scripts\llm_connection_test\check_llm_connection.py
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
