from pymongo import MongoClient
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

client = MongoClient("mongodb://localhost:27017/")
collection = client['attractions']['user_data']

for doc in collection.find():
    producer.send('attractions_topic', value=doc)
    print(f"Produced: {doc}")
    time.sleep(1)  # Simulate real-time stream
