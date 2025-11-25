from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, date_format
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Spark session
spark = SparkSession.builder \
    .appName("kafka_to_hdfs") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("node_id", StringType(), True),
    StructField("consumption_kw", DoubleType(), True),
])

kafka_df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "electricity_usage") \
    .option("startingOffsets", "latest") \
    .load()

json_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str")
parsed = json_df.select(from_json(col("json_str"), schema).alias("data")) \
    .select(
        to_timestamp(col("data.timestamp")).alias("ts"),
        col("data.node_id").alias("node_id"),
        col("data.consumption_kw").alias("consumption_kw")
    )

out_df = parsed.withColumn("date", date_format(col("ts"), "yyyy-MM-dd")) \
               .withColumn("hour", date_format(col("ts"), "HH"))

# Use explicit HDFS URIs (avoid uri/authority ambiguity)
output_path = "hdfs://localhost:9000/raw_data"
checkpoint_path = "hdfs://localhost:9000/checkpoints/kafka_stream"

query = out_df.writeStream \
    .format("parquet") \
    .option("path", output_path) \
    .option("checkpointLocation", checkpoint_path) \
    .partitionBy("date", "node_id") \
    .outputMode("append") \
    .start()

print("Started spark structured streaming job. Waiting for data...")
query.awaitTermination()
