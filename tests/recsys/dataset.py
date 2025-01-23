from recsys.datasets import YelpDestinationsGNNDataset, YelpUserGNNDataset, YelpReviewsGNNDataset


dataset = YelpDestinationsGNNDataset()

dataset.read_csv("./data/places.csv")

dataset.process()

print(dataset.df)


dataset = YelpUserGNNDataset()

dataset.read_csv("./data/users.csv")

dataset.process()

print(dataset.df)


dataset = YelpReviewsGNNDataset()

dataset.read_csv("./data/reviews.csv")

dataset.process()

print(dataset.df)