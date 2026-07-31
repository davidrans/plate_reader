FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# curl is needed for Coolify's container-internal health check (it execs
# curl/wget inside the container) — bookworm-slim doesn't include either.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock* /app/
RUN uv sync --no-install-project

COPY . /app
RUN uv sync

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
