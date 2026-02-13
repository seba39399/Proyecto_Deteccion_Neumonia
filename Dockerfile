# Imagen base de Python
FROM python:3.10-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-cache

COPY . .

ENV DISPLAY=host.docker.internal:0.0

CMD ["uv", "run", "main.py"]