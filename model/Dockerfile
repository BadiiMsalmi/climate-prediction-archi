# ---------- BUILDER STAGE ----------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------- FINAL IMAGE ----------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

COPY . .

ENV PGUSER=badii
ENV PGPASS=airpass
ENV PGHOST=postgres
ENV PGPORT=5432
ENV PGDB=temperature_db

RUN mkdir -p /app/saved_models

CMD ["tail", "-f", "/dev/null"]

