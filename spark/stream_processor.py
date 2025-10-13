import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

spark = (
    SparkSession.builder
    .appName("WeatherStreamProcessor")
    .config("spark.jars", "/opt/spark/jars/postgresql-42.7.3.jar")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "default_topic")
PG_HOST = "postgres" # The service name
PG_PORT = "5432" # The standard port
PG_DB = os.getenv("POSTGRES_DB", "temperature_db") # Use the DB name

jdbc_url = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}"
PG_USER = os.getenv("POSTGRES_USER", "default_user")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "default_password")

schema = StructType([
    StructField("timestamp", StringType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType())
])

# Read from Kafka
df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKER)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

json_df = df.selectExpr("CAST(value AS STRING)")
data_df = json_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Clean data
clean_df = (
    data_df
    .filter(
        (col("temperature").isNotNull()) &
        (col("humidity").isNotNull()) &
        (col("latitude").isNotNull()) &
        (col("longitude").isNotNull()) &
        (col("timestamp").isNotNull())
    )
    .filter(
        (col("temperature") >= -50) & (col("temperature") <= 60) &
        (col("humidity") >= 0) & (col("humidity") <= 100)
    )
    .withColumn("timestamp", to_timestamp(col("timestamp")))
)

print(clean_df)
# Write each micro-batch to Postgres
def write_to_postgres(batch_df, batch_id):
    print(f"\n=== Batch {batch_id} ===")
    batch_df.show(truncate=False)

    (
        batch_df.write
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", "temperature_hourly")
        .option("user", PG_USER)
        .option("password", PG_PASSWORD)
        .option("driver", "org.postgresql.Driver")
        .mode("append")
        .save()
    )

query = (
    clean_df.writeStream
    .foreachBatch(write_to_postgres)
    .outputMode("append")
    .option("checkpointLocation", "/tmp/spark_checkpoint")
    .start()
)

query.awaitTermination()
