FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PGUSER=badii
ENV PGPASS=airpass
ENV PGHOST=postgres
ENV PGPORT=5432
ENV PGDB=temperature_db

RUN mkdir -p /app/saved_models

CMD ["python", "initial_train.py"]