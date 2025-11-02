#producer.py
import os
import json
import time
import requests
from kafka import KafkaProducer

API_URL = os.getenv(
    "API_URL",
    "default_url"
)
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "default_broker")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "weather_data")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

while True:
    print("Fetching data from Open-Meteo API...")
    try:
        response = requests.get(API_URL)
        data = response.json()

        current = data.get("current", {})
        payload = {
            "timestamp": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude")
        }

        producer.send(KAFKA_TOPIC, value=payload)
        print("Sent:", payload)

    except Exception as e:
        print(f"Error fetching data: {e}")

    time.sleep(3600)
