import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, arrays_zip, to_timestamp, lit

spark = SparkSession.builder.appName("LoadHistoricalWeatherData").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

file_path = "tunisia_5y.json"
df = spark.read.option("multiline", "true").json(file_path)

# Extract static values (since there’s only one JSON object)
latitude = df.select("latitude").first()[0]
longitude = df.select("longitude").first()[0]

# Zip time & temperature arrays
df_zipped = df.select(arrays_zip(col("hourly.time"), col("hourly.temperature_2m"), col("hourly.relative_humidity_2m")).alias("hourly_data"))

# Explode and reattach static latitude/longitude as literal columns
df_exploded = df_zipped.select(
    explode(col("hourly_data")).alias("hour")
).withColumn("latitude", lit(latitude)).withColumn("longitude", lit(longitude))

# Clean and format
df_cleaned = df_exploded.select(
    to_timestamp(col("hour.time")).alias("timestamp"),
    col("hour.temperature_2m").alias("temperature"),
    col("hour.relative_humidity_2m").alias("humidity"),
    col("latitude"),
    col("longitude")
)


df_cleaned.show(5)

PG_USER = os.getenv("POSTGRES_USER", "default_user")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "default_password")
PG_DB = os.getenv("POSTGRES_DB", "default_db")

# Write to PostgreSQL
jdbc_url = f"jdbc:postgresql://postgres:5432/{PG_DB}"
table_name = "temperature_hourly"
db_properties = {
    "user": PG_USER,
    "password": PG_PASSWORD,
    "driver": "org.postgresql.Driver"
}

df_cleaned.write.jdbc(url=jdbc_url, table=table_name, mode="append", properties=db_properties)
print("✅ Historical weather data successfully loaded into PostgreSQL.")
