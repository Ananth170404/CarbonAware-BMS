from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, avg, lag, unix_timestamp, from_unixtime, hour, dayofweek

spark = SparkSession.builder.appName("feature_engineering").getOrCreate()

print("=== Reading raw data from HDFS ===")
df = spark.read.parquet("hdfs://localhost:9000/raw_data/")
print("Total raw records:", df.count())

# convert ts to hourly bucket (rounded down)
df_hour = df.withColumn(
    "hour_ts",
    from_unixtime((unix_timestamp(col("ts")) / 3600).cast("long") * 3600)
).groupBy("node_id", "hour_ts").agg(avg("consumption_kw").alias("consumption_kw"))

print("After hourly aggregation:", df_hour.count())

# define lag and rolling windows
w = Window.partitionBy("node_id").orderBy("hour_ts")
df_feat = (
    df_hour
    .withColumn("lag_1", lag("consumption_kw", 1).over(w))
    .withColumn("lag_24", lag("consumption_kw", 24).over(w))
    .withColumn("lag_168", lag("consumption_kw", 168).over(w))
)

# rolling mean of last 24 hours
w_24 = Window.partitionBy("node_id").orderBy("hour_ts").rowsBetween(-23, 0)
df_feat = df_feat.withColumn("rolling_24", avg("consumption_kw").over(w_24))

# add time features
df_feat = df_feat.withColumn("hour", hour(col("hour_ts"))).withColumn("dayofweek", dayofweek(col("hour_ts")))

# drop only rows where consumption_kw is null (keep lag nulls)
df_final = df_feat.filter(col("consumption_kw").isNotNull())

print("Final record count:", df_final.count())

# write to HDFS
output_path = "hdfs://localhost:9000/processed_data/features_hourly/"
df_final.write.mode("overwrite").partitionBy("node_id").parquet(output_path)
print(f"=== Features written successfully to {output_path} ===")

