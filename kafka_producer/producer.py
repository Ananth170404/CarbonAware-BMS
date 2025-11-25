# kafka_producer/producer.py
import json
import time
import pandas as pd
from kafka import KafkaProducer

# Kafka config
BOOTSTRAP = "localhost:9092"
TOPIC = "electricity_usage"

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=10
)

# Load dataset
df = pd.read_csv("bigdata_preprocess.csv")

# Preview dataset
print("Dataset preview:")
print(df.head())

# Function to send messages
def send_messages():
    count = 0
    for _, row in df.iterrows():
        timestamp = str(row["date_time"])   # <-- use correct column
        for col in df.columns[1:]:          # skip 'date_time', loop over MT_001 ... MT_100
            msg = {
                "timestamp": timestamp,
                "node_id": col,
                "consumption_kw": float(row[col])
            }
            producer.send(TOPIC, msg)
            count += 1
    producer.flush()
    print(f"Sent batch of {count} messages to Kafka.")

if __name__ == "__main__":
    while True:
        send_messages()
        time.sleep(10)   # for testing; change to 900 (15 min) in real world
