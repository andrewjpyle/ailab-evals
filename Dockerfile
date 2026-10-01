FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY suites ./suites
COPY fixtures ./fixtures
COPY results ./results
ENV PATH="/app/.venv/bin:$PATH"
# Replays the committed cassettes: no network, no API key needed.
RUN useradd --create-home --uid 10001 evals && chown -R evals /app
USER evals
ENTRYPOINT ["ailab-evals"]
CMD ["run", "suites/substance.json", "--out", "/tmp/out"]
