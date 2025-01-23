from recsys.datasets import (
    YelpDestinationsGNNDataset, YelpUserGNNDataset, YelpReviewsGNNDataset,
    YelpGNNDataset
    )


places = YelpDestinationsGNNDataset()
places.read_csv("./data/places.csv")
places.process()


users = YelpUserGNNDataset()
users.read_csv("./data/users.csv")
users.process()



reviews = YelpReviewsGNNDataset()
reviews.read_csv("./data/reviews.csv")
reviews.process()



gnn_dataset = YelpGNNDataset(
    user_df= users.df, destinations_df= places.df, reviews_df= reviews.df
)
gnn_dataset.process()

print(gnn_dataset.get_gnn_datasets())