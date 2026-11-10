# Model service connectivity

Use CMD. Activate the project's single virtual environment once in every new
terminal, then use short Python commands:

```cmd
cd /d D:\my_coach\new_agent\ResearchClaw
.venv\Scripts\activate
python scripts/llm_connection_test/check_llm_connection.py
```

Create `.venv` with `python -m venv .venv` first if it does not exist. After
activation, run `python -m pip install -r requirements.txt` only for initial setup
or dependency changes. Use `deactivate` to leave the environment.

Optional arguments:

```text
--timeout SECONDS      Per-attempt timeout (default: 30)
--max-attempts COUNT   Attempts for retryable failures (default: 3)
--json                 Machine-readable output
```

Exit codes are `0` for success, `1` for a network/provider failure, `2` for
invalid configuration or arguments, and `3` for an unexpected model response.
The script never prints `RESEARCHCLAW_MODEL_API_KEY`.
