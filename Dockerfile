FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY classification/pyproject.toml classification/uv.lock ./classification/

RUN cd classification && uv sync --frozen

COPY . .

EXPOSE 8000

# Default command
CMD ["uv", "run", "--project", "classification", "python", "commands.py", "serve", "--host", "0.0.0.0"]
