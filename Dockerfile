# Phonebot Test Platform — orchestrator image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PHONEBOT_SCENARIOS_DIR=/app/scenarios \
    PHONEBOT_PERSONAS_DIR=/app/personas

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY phonebot_qa ./phonebot_qa
RUN pip install --upgrade pip && pip install ".[api]"

# Scenarios & personas are data (the source of truth) — copy them in.
COPY scenarios ./scenarios
COPY personas ./personas

EXPOSE 8000

# Default: serve the orchestrator API. Override the command to run the CLI, e.g.
#   docker run --rm phonebot-qa phonebot-qa run --suite all
CMD ["phonebot-qa", "serve", "--host", "0.0.0.0", "--port", "8000"]
