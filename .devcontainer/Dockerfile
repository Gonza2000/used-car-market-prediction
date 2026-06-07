FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*
RUN git lfs install --system
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
RUN uv venv Gonzalo
COPY requirements.txt .
RUN uv pip install --python Gonzalo/bin/python -r requirements.txt
ENV VIRTUAL_ENV=/app/Gonzalo
ENV PATH="/app/Gonzalo/bin:$PATH"
