FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
WORKDIR /app
RUN python -m venv /opt/venv
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY backend/src/ backend/src/
RUN pip install --no-cache-dir -e ".[web,pdf]"
COPY config/ config/
COPY skills/ skills/
EXPOSE 8000
CMD ["uvicorn", "researchops_api.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
