from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, FloatType

spark = SparkSession.builder \
    .appName("KafkaStreamProcessor") \
    .master("local[*]") \
    .getOrCreate()

schema = StructType() \
    .add("user_id", StringType()) \
    .add("location_id", StringType()) \
    .add("ratings", FloatType()) \
    .add("location_name", StringType())

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "attractions_topic") \
    .load()

df = df.selectExpr("CAST(value AS STRING)")
json_df = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Process recommendations (dummy logic for illustration)
recommendations = json_df.groupBy("location_name").avg("ratings")

query = recommendations.writeStream \
    .outputMode("complete") \
    .format("console") \
    .start()

query.awaitTermination()
