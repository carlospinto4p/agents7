FROM python:3.12-slim

ARG GITLAB_TOKEN

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN UV_INDEX_GITLAB_CPINTO_USERNAME=oauth2 \
    UV_INDEX_GITLAB_CPINTO_PASSWORD=${GITLAB_TOKEN} \
    uv pip install --system ".[ui]"

EXPOSE 8501

CMD ["streamlit", "run", "src/agents7/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
