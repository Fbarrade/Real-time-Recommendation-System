from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['tourism_recommendations']  # Your database name

# Create Places collection with schema validation
db.create_collection("Places", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "category", "address", "location", "opening_hours"],
        "properties": {
            "name": {
                "bsonType": "string",
                "description": "Name of the place is required and must be a string."
            },
            "category": {
                "enum": ["restaurant", "hotel", "attraction"],
                "description": "Category must be one of 'restaurant', 'hotel', or 'attraction'."
            },
            "description": {
                "bsonType": "string",
                "description": "Optional description of the place."
            },
            "address": {
                "bsonType": "object",
                "required": ["street", "city", "country", "postal_code"],
                "properties": {
                    "street": {
                        "bsonType": "string",
                        "description": "Street address is required and must be a string."
                    },
                    "city": {
                        "bsonType": "string",
                        "description": "City is required and must be a string."
                    },
                    "country": {
                        "bsonType": "string",
                        "description": "Country is required and must be a string."
                    },
                    "postal_code": {
                        "bsonType": "string",
                        "description": "Postal code is required and must be a string."
                    }
                }
            },
            "location": {
                "bsonType": "object",
                "required": ["lat", "lon"],
                "properties": {
                    "lat": {
                        "bsonType": "double",
                        "description": "Latitude is required and must be a double."
                    },
                    "lon": {
                        "bsonType": "double",
                        "description": "Longitude is required and must be a double."
                    }
                }
            },
            "tags": {
                "bsonType": "array",
                "items": {
                    "bsonType": "string"
                },
                "description": "Optional array of tags for the place."
            },
            "average_rating": {
                "bsonType": "double",
                "description": "Optional average rating of the place."
            },
            "sentiment_score": {
                "bsonType": "double",
                "description": "Optional sentiment score based on reviews."
            },
            "reviews_count": {
                "bsonType": "int",
                "description": "Optional count of total reviews for the place."
            },
            "opening_hours": {
                "bsonType": "object",
                "additionalProperties": {
                    "bsonType": "string",
                    "description": "Opening hours for each day of the week in 'HH:mm-HH:mm' format."
                },
                "description": "Opening hours are required and must be an object."
            },
            "preferred_visit_period": {
                "bsonType": "array",
                "items": {
                    "bsonType": "string"
                },
                "description": "Optional array of preferred months to visit the place."
            },
            "popular_visit_times": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "properties": {
                        "day": {
                            "bsonType": "string",
                            "description": "Day of the week."
                        },
                        "hours": {
                            "bsonType": "array",
                            "items": {
                                "bsonType": "string"
                            },
                            "description": "Array of time ranges for popular visit times."
                        }
                    }
                },
                "description": "Optional popular visit times."
            },
            "entry_fee": {
                "bsonType": "object",
                "properties": {
                    "adult": {
                        "bsonType": "double",
                        "description": "Entry fee for adults."
                    },
                    "child": {
                        "bsonType": "double",
                        "description": "Entry fee for children."
                    },
                    "currency": {
                        "bsonType": "string",
                        "description": "Currency of the entry fee."
                    }
                },
                "description": "Optional entry fee information."
            },
            "contact_info": {
                "bsonType": "object",
                "properties": {
                    "phone": {
                        "bsonType": "string",
                        "description": "Phone number for the place."
                    },
                    "website": {
                        "bsonType": "string",
                        "description": "Website URL for the place."
                    }
                },
                "description": "Optional contact information."
            },
            "photos": {
                "bsonType": "array",
                "items": {
                    "bsonType": "string"
                },
                "description": "Optional array of photo URLs for the place."
            },
            "special_events": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "properties": {
                        "event_name": {
                            "bsonType": "string",
                            "description": "Name of the special event."
                        },
                        "event_date": {
                            "bsonType": "string",
                            "description": "Date and time of the event in ISO format."
                        },
                        "description": {
                            "bsonType": "string",
                            "description": "Description of the event."
                        }
                    }
                },
                "description": "Optional array of special events."
            },
            "recent_reviews": {
                "bsonType": "array",
                "items": {
                    "bsonType": "object",
                    "properties": {
                        "review_id": {
                            "bsonType": "string",
                            "description": "Unique identifier for the review."
                        },
                        "sentiment": {
                            "bsonType": "double",
                            "description": "Sentiment score of the review."
                        },
                        "timestamp": {
                            "bsonType": "string",
                            "description": "Timestamp of the review in ISO format."
                        }
                    }
                },
                "description": "Optional array of recent reviews."
            }
        }
    }
})

# Create Users collection with schema validation
db.create_collection("Users", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "username", "email", "location", "preferences"],
        "properties": {
            "user_id": {
                "bsonType": "string",
                "description": "User's unique identifier."
            },
            "username": {
                "bsonType": "string",
                "description": "Username of the user."
            },
            "email": {
                "bsonType": "string",
                "description": "Email address of the user."
            },
            "location": {
                "bsonType": "object",
                "required": ["lat", "lon"],
                "properties": {
                    "lat": {
                        "bsonType": "double",
                        "description": "Latitude of the user's location."
                    },
                    "lon": {
                        "bsonType": "double",
                        "description": "Longitude of the user's location."
                    }
                }
            },
            "preferences": {
                "bsonType": "object",
                "properties": {
                    "categories": {
                        "bsonType": "array",
                        "items": {
                            "bsonType": "string"
                        },
                        "description": "User's preferred categories for recommendations."
                    },
                    "budget_range": {
                        "bsonType": "object",
                        "properties": {
                            "min": {
                                "bsonType": "double",
                                "description": "User's minimum budget."
                            },
                            "max": {
                                "bsonType": "double",
                                "description": "User's maximum budget."
                            }
                        },
                        "description": "User's budget range."
                    }
                }
            },
            "reviews": {
                "bsonType": "array",
                "items": {
                    "bsonType": "string"
                },
                "description": "Array of review IDs the user has written."
            }
        }
    }
})

# Create Reviews collection with schema validation
db.create_collection("Reviews", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["review_id", "user_id", "place_id", "rating", "comment", "timestamp"],
        "properties": {
            "review_id": {
                "bsonType": "string",
                "description": "Unique identifier for the review."
            },
            "user_id": {
                "bsonType": "string",
                "description": "User ID of the reviewer."
            },
            "place_id": {
                "bsonType": "string",
                "description": "Place ID the review is related to."
            },
            "rating": {
                "bsonType": "double",
                "description": "Rating given by the user."
            },
            "comment": {
                "bsonType": "string",
                "description": "Comment or feedback from the user."
            },
            "timestamp": {
                "bsonType": "string",
                "description": "Timestamp of the review in ISO format."
            }
        }
    }
})

# Create Bookmarks collection with schema validation
db.create_collection("Bookmarks", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "place_id", "timestamp"],
        "properties": {
            "user_id": {
                "bsonType": "string",
                "description": "User ID of the person who bookmarked the place."
            },
            "place_id": {
                "bsonType": "string",
                "description": "Place ID of the bookmarked place."
            },
            "timestamp": {
                "bsonType": "string",
                "description": "Timestamp of when the place was bookmarked in ISO format."
            }
        }
    }
})

# Insert example data into Places, Users, Reviews, Bookmarks
places_collection = db["Places"]
users_collection = db["Users"]
reviews_collection = db["Reviews"]
bookmarks_collection = db["Bookmarks"]

# Example document for Places
place_example = {
    "name": "Eiffel Tower",
    "category": "attraction",
    "address": {
        "street": "Champ de Mars",
        "city": "Paris",
        "country": "France",
        "postal_code": "75007"
    },
    "location": {
        "lat": 48.8584,
        "lon": 2.2945
    },
    "tags": ["landmark", "tourist attraction"],
    "average_rating": 4.7,
    "sentiment_score": 0.85,
    "reviews_count": 15000,
    "opening_hours": {
        "Monday": "09:00-23:00",
        "Tuesday": "09:00-23:00",
        "Wednesday": "09:00-23:00",
        "Thursday": "09:00-23:00",
        "Friday": "09:00-23:00",
        "Saturday": "09:00-23:00",
        "Sunday": "09:00-23:00"
    },
    "preferred_visit_period": ["April", "May", "June"],
    "popular_visit_times": [
        {"day": "Saturday", "hours": ["10:00-12:00", "14:00-16:00"]},
        {"day": "Sunday", "hours": ["09:00-11:00", "15:00-17:00"]}
    ],
    "entry_fee": {
        "adult": 25.0,
        "child": 15.0,
        "currency": "EUR"
    },
    "contact_info": {
        "phone": "+33 1 72 56 47 02",
        "website": "https://www.toureiffel.paris"
    },
    "photos": ["https://example.com/eiffel_tower.jpg"],
    "special_events": [
        {
            "event_name": "New Year's Eve Celebration",
            "event_date": "2025-12-31T21:00:00",
            "description": "Celebrate the New Year with a special event at the Eiffel Tower."
        }
    ],
    "recent_reviews": [
        {
            "review_id": "rev12345",
            "sentiment": 0.75,
            "timestamp": "2025-01-14T12:00:00"
        }
    ]
}

places_collection.insert_one(place_example)

# Example document for Users
user_example = {
    "user_id": "user123",
    "username": "john_doe",
    "email": "johndoe@example.com",
    "location": {
        "lat": 48.8566,
        "lon": 2.3522
    },
    "preferences": {
        "categories": ["restaurant", "attraction"],
        "budget_range": {
            "min": 20.0,
            "max": 100.0
        }
    },
    "reviews": ["rev12345"]
}

users_collection.insert_one(user_example)

# Example document for Reviews
review_example = {
    "review_id": "rev12345",
    "user_id": "user123",
    "place_id": "place123",
    "rating": 4.5,
    "comment": "Amazing view, worth the visit!",
    "timestamp": datetime.now().isoformat()
}

reviews_collection.insert_one(review_example)

# Example document for Bookmarks
bookmark_example = {
    "user_id": "user123",
    "place_id": "place123",
    "timestamp": datetime.now().isoformat()
}

bookmarks_collection.insert_one(bookmark_example)

print("Collections created and example data inserted successfully.")
