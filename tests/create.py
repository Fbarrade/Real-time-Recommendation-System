import pandas as pd
import json 

user_df = pd.read_csv("./data/users.csv")
reviews_df = pd.read_csv("./data/reviews.csv")
places_df = pd.read_csv("./data/places.csv")

print(user_df, reviews_df, places_df)


with open("./data/predict-file.json", "w") as f:
    
    data = {
        "users": user_df.iloc[1:3, :].to_dict(),
        "places": places_df.iloc[1:3, :].to_dict(),
        "reviews": reviews_df.iloc[1:3, :].to_dict()
    }

    json.dump(data, f)