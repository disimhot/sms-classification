FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY classification/pyproject.toml classification/uv.lock ./

RUN uv sync --frozen

COPY . .

# Default command
CMD ["uv", "run", "python", "commands.py", "train"]
