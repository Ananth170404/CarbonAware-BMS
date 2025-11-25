from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, date_format, col
import sys

# INPUT_PATH can be local or HDFS path
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/opt/electricity-pipeline/bigdata_preprocess.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "hdfs:///raw_data/"
APP_NAME = "csv_to_parquet"

spark = SparkSession.builder.appName(APP_NAME).getOrCreate()

# 1) Read CSV (header present). Infer schema disabled to keep types safe; we'll cast later.
df = spark.read.option("header", "true").csv(INPUT_PATH)

# 2) Identify meter columns (all except the timestamp column)
time_col = "date_time"
meter_cols = [c for c in df.columns if c != time_col]

if not meter_cols:
    raise RuntimeError("No meter columns found. Check CSV header. Found: {}".format(df.columns))

# 3) Use stack to unpivot wide -> long
n = len(meter_cols)
pairs = ", ".join([f"'{c}', `{c}`" for c in meter_cols])
stack_expr = f"stack({n}, {pairs}) as (node_id, consumption_kw)"

long_df = df.selectExpr(f"`{time_col}` as date_time", stack_expr)

# 4) Cast timestamp and consumption types
long_df = (long_df.withColumn("ts", to_timestamp(col("date_time"), "yyyy-MM-dd HH:mm:ss"))
                     .withColumn("consumption_kw", col("consumption_kw").cast("double"))
                     .drop("date_time"))

# 5) Add partition columns
long_df = (long_df.withColumn("date", date_format(col("ts"), "yyyy-MM-dd"))
                    .withColumn("hour", date_format(col("ts"), "HH")))

# 6) Basic validations (drop nulls)
clean = long_df.filter(col("node_id").isNotNull() & col("consumption_kw").isNotNull())

# 7) Write to HDFS as Parquet partitioned by date and node_id
clean.write.mode("append").partitionBy("date", "node_id").parquet(OUT_PATH)

spark.stop()
