
from utils.Recommender_System import RecommendationSystem
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

# Main execution
if __name__ == "__main__":
    spark = SparkSession.builder.appName("RecommendationSystem").getOrCreate()

    recommender = RecommendationSystem(spark, "ratings_test.csv")
    recommender.load_data()
    recommender.split_data()
    recommender.train_model()
    recommender.evaluate_model()

    # Optimize the model
    recommender.optimize_model()

    # Generate recommendations
    user_recs, item_recs = recommender.generate_recommendations()

    # Display recommendations
    user_recs.show(truncate=False)
    item_recs.show(truncate=False)

    # Save the model
    recommender.save_model("als_model")
    
    spark.stop()