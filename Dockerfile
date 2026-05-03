# syntax=docker/dockerfile:1
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

# The token is consumed here and is not persisted in any image layer
# (BuildKit secret mount) nor exported to the runtime environment.
RUN --mount=type=secret,id=gitlab_token \
    UV_INDEX_GITLAB_CPINTO_USERNAME=oauth2 \
    UV_INDEX_GITLAB_CPINTO_PASSWORD=$(cat /run/secrets/gitlab_token) \
    uv pip install --system ".[ui]"

EXPOSE 8501

CMD ["streamlit", "run", "src/agents7/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
