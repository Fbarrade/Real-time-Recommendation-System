
from pymongo import MongoClient

data = [
    {"user_id": "u1", "location_id": "l1", "ratings": 4.5, "location_name": "Park A"},
    {"user_id": "u2", "location_id": "l2", "ratings": 3.8, "location_name": "Museum B"},
    # Add more data points...
]

client = MongoClient("mongodb://localhost:27017/")
db = client['attractions']
collection = db['user_data']
collection.insert_many(data)
print("Data loaded successfully!")
