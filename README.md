# 🌤️ Tunisia Climate Data Pipeline (Spark, Kafka, PostgreSQL, DL/LSTM)

## 🧠 Overview

This project builds an **end-to-end climate data pipeline** for **real-time and historical weather processing** and **temperature forecasting** in **Tunisia**.
It integrates **data engineering** and **AI** components into one Dockerized architecture.

The system includes:

1. **Historical Data Loader (Spark):** Loads multi-year historical weather data into PostgreSQL.
2. **Real-Time Data Producer (Python + Kafka):** Periodically fetches live weather metrics from Open-Meteo API and streams them to Kafka.
3. **Stream Processor (Spark Structured Streaming):** Consumes live Kafka data, cleans it, validates it, and stores it in PostgreSQL.
4. **PostgreSQL Database:** Central storage for both historical and real-time datasets.
5. **Model Trainer (TensorFlow):** Periodically retrains an LSTM model on stored data and saves updated models.
6. **Model API (Flask):** Provides REST endpoints to serve **live temperature predictions** using the most recently trained model.

Everything runs in isolated **Docker containers** for full reproducibility.

---

## 🏗️ Architecture

```
            +------------------+
            |  Open-Meteo API  |
            +--------+---------+
                     |
                     ▼
             +---------------+
             |  Producer.py   |
             | (Python + API) |
             +-------+--------+
                     |
                     ▼
          +--------------------+
          |   Kafka (Broker)   |
          +---------+----------+
                    |
                    ▼
        +---------------------------+
        |  Spark Stream Processor   |
        | (Clean + Transform)       |
        +-------------+-------------+
                      |
                      ▼
           +-----------------------+
           |   PostgreSQL Database |
           +-----------+-----------+
                       |
             ┌─────────┴─────────┐
             ▼                   ▼
  +------------------+  +-------------------+
  |  Model Trainer   |  |   Model API       |
  | (LSTM, TF/Keras) |  | (Flask, REST)     |
  +------------------+  +-------------------+
             |
             ▼
      Saved Models (/app/saved_models)
```

---

## ⚙️ Components

### 🧩 **1. Producer Service**

* **Path:** `/producer/producer.py`
* Fetches data from [Open-Meteo API](https://open-meteo.com/) every hour.
* Sends messages like:

  ```json
  {
    "timestamp": "2025-11-02T14:00",
    "temperature": 24.6,
    "humidity": 55.0,
    "latitude": 36.81,
    "longitude": 10.19
  }
  ```
* Sends them to Kafka topic `weather_data`.

---

### ⚡ **2. Kafka + Zookeeper**

* Manages message queues for real-time data.
* **Service names:** `kafka`, `zookeeper`
* Internal network communication through `weather_data_net`.
* Exposed on port **9092**.

---

### 🔥 **3. Spark Streaming Processor**

* **Path:** `/spark/stream_processor.py`
* Reads from Kafka topic (`weather_data`).
* Cleans and validates incoming JSON data.
* Writes each micro-batch to PostgreSQL.
* Example cleaning rules:

  * Drop rows with missing temperature/humidity.
  * Accept temperature in [-50, 60] °C and humidity [0, 100]%.

---

### 🧱 **4. PostgreSQL Database**

* **Service name:** `postgres`
* Stores:

  * Historical data (`temperature_hourly`)
  * Model training data
  * Live predictions

---

### 🧠 **5. Model Trainer**

* **Path:** `/model/train_model.py` (check repo)
* Reads processed data from PostgreSQL.
* Trains an **LSTM** model to predict next-hour temperature.
* Saves:

  * `model.h5` (trained model)
  * `scaler_X.pkl` and `scaler_y.pkl` (data scalers)
  * `current_model.json` (pointer to latest model)
* Output directory: `/model/saved_models/`

---

### 🔮 **6. Model API**

* **Path:** `/model_api/model_api.py`
* Flask-based REST API serving predictions from latest LSTM model.
* Reads last 24 hours of weather data from PostgreSQL.
* Transforms them into model input format.
* Returns JSON:

  ```json
  {
    "predicted_temperature_next_hour": 25.72
  }
  ```
* Endpoint:

  ```
  GET http://localhost:8000/predict
  ```
* Automatically logs predictions in the `predictions` table.

---

## ⚡ Usage

### 1️⃣ **Setup**

Clone the repo:

```bash
git clone https://github.com/BadiiMsalmi/climate-prediction-archi.git
cd climate-prediction-archi
```

Create your `.env` file in the root directory:

```ini
# PostgreSQL
POSTGRES_USER=badii
POSTGRES_PASSWORD=
POSTGRES_DB=temperature_db

# Kafka
ZOOKEEPER_CLIENT_PORT=2181
KAFKA_BROKER_ID=1
KAFKA_TOPIC=weather_data
KAFKA_PORT=9092

# API
API_URL=https://api.open-meteo.com/v1/forecast?latitude=36.801403&longitude=10.170828&current=temperature_2m,relative_humidity_2m
```

---

### 2️⃣ **Run All Containers**

```bash
docker compose up --build -d
```

This starts:

* Zookeeper
* Kafka
* PostgreSQL
* Spark Master & Worker
* Producer
* Model Trainer
* Model API (Flask)

---

### 3️⃣ **Load Historical Data**

Enter the Spark Master container:

```bash
docker exec -it spark-master /bin/bash
```

Run:

```bash
/opt/spark/bin/spark-submit /app/load_historical.py
```

---

### 4️⃣ **Start Stream Processing**

Inside the same container:

```bash
/opt/spark/bin/spark-submit /opt/spark/app/stream_processor.py
```

You’ll see micro-batches appearing in logs:

```bash
=== Batch 1 ===
+-------------------+-----------+---------+-----------+-----------+
|timestamp          |temperature|humidity |latitude   |longitude  |
+-------------------+-----------+---------+-----------+-----------+
|2025-11-02 14:30:00|25.7       |40.0     |36.8125    |10.1875    |
+-------------------+-----------+---------+-----------+-----------+
```

---

### 5️⃣ **Access the Model API**

Predict next-hour temperature:

```bash
curl http://localhost:8000/predict
```

Response:

```json
{"predicted_temperature_next_hour": 25.72}
```

---

### 6️⃣ **Shut Down**

Stop and clean up all services:

```bash
docker compose down
```

---
## 📈 Future Improvements

✅ Add Grafana dashboard for temperature trends

