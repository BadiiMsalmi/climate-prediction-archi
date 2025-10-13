````markdown
# 🇹🇳 Tunisia Climate Data Pipeline (Spark, Kafka, PostgreSQL, DL/LSTM)

## 📝 Project Description

This project establishes a complete data engineering pipeline for ingesting, processing, and storing Tunisian climate data, ultimately feeding a Deep Learning model for temperature prediction.

The pipeline comprises the following key components, all running in Docker containers:

1. **Historical Data Load (Spark):** Loads 5 years (2020-2025) of historical climate data from a JSON file into the PostgreSQL database.
2. **Real-Time Producer (Python/Kafka):** A Python service that hits the Open-Meteo API every hour to get real-time temperature data for Tunisia. It produces these data points to a Kafka topic.
3. **Spark Streaming Processor:** Consumes the real-time data from Kafka, performs cleaning and validation, and persists the clean records to the PostgreSQL database, appending to the historical data.
4. **PostgreSQL Database:** Acts as the unified data store for both historical and real-time climate observations.
5. **Deep Learning (LSTM):** The final consumer that will use the combined data for training a time-series prediction model (LSTM) to forecast future temperatures.

## 🚀 How to Use

Follow these steps to set up and run the entire containerized pipeline:

### 1. Prerequisites

You must have the following installed on your machine:

  * **Docker**
  * **Docker Compose** (or Docker Desktop)

### 2. Setup and Configuration

1. **Clone the Repository:**

    ```bash
    git clone [Your-Repo-URL]
    cd [your-project-directory]
    ```

2. **Create and Configure `.env`:**
    Since the `.env` file is intentionally omitted from the repository for security, you must create it manually in the root directory. This file contains private secrets (passwords and API configurations).

    Create a file named **`.env`** and populate it with your environment variables (using a strong password):

    ```ini
    # .env (DO NOT COMMIT THIS FILE!)

    # PostgreSQL Credentials
    POSTGRES_USER=badii
    POSTGRES_PASSWORD=your_strong_unique_password # <--- CHANGE ME
    POSTGRES_DB=temperature_db

    # Kafka/Network Configuration
    ZOOKEEPER_CLIENT_PORT=2181
    KAFKA_BROKER_ID=1
    KAFKA_TOPIC=weather_data
    KAFKA_PORT=9092

    # API Configuration (Open-Meteo URL)
    API_URL=https://api.open-meteo.com/v1/forecast?latitude=36.801403&longitude=10.170828&current=temperature_2m,relative_humidity_2m
    ```

### 3. Running the Pipeline

Use Docker Compose to build and start all services (Zookeeper, Kafka, PostgreSQL, Spark, and the Producer):

```bash
docker compose up --build -d
````

### 4. Initial Historical Data Load

Once the containers are running, you must run your script to load the historical JSON data into the PostgreSQL database:

* **Note:** Assuming your historical script is run from inside the `spark-master` container.

  ```bash
  # Connect to the Spark Master container
  docker exec -it spark-master /bin/bash

  # Inside the container, run your historical load script (replace 'load_historical.py' with your script name)
  /opt/spark/bin/spark-submit /app/load_historical.py

  # Exit the container
  exit
  ```

### 5. Verify the Stream

Start with : 
```bash
docker exec -it spark-master /bin/bash
```

then run this command : 
```bash
/opt/spark/bin/spark-submit /opt/spark/app/stream_processor.py
```

* **Real-Time Data:** The `producer` container is now fetching new temperature data every hour and sending it to the `kafka` broker.
* **Data Processing:** The `spark-master` container is running the stream processor, which continuously reads from Kafka and writes clean data to the `postgres` database.

You can check the logs of the Spark Streaming container to confirm data flow:

```bash
docker logs -f spark-master
```
Also you can check with SELECT COUNT(*) FROM TABLE_NAME; that the count is increasing.

### 6. Shut Down

To stop and remove all containers, networks, and volumes (excluding named volumes):

```bash
docker compose down
```

