from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

class RecommendationSystem:
    def __init__(self, spark_session, data_path):
        self.spark = spark_session
        self.data_path = data_path
        self.model = None

    def load_data(self):
        """Load data from the specified path."""
        self.ratings = self.spark.read.options(header=True, inferSchema=True).csv(self.data_path)

    def split_data(self):
        """Split data into training and test sets."""
        self.training, self.test = self.ratings.randomSplit([0.8, 0.2], seed=42)

    def train_model(self, max_iter=5, reg_param=0.01):
        """Train an ALS model."""
        als = ALS(maxIter=max_iter, regParam=reg_param, userCol="user_id", itemCol="location_id", 
                  ratingCol="ratings", coldStartStrategy="drop")
        self.model = als.fit(self.training)

    def evaluate_model(self):
        """Evaluate the model using RMSE."""
        predictions = self.model.transform(self.test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print(f"Root-mean-square error (RMSE): {rmse}")

    def optimize_model(self):
        """Optimize model using cross-validation."""
        als = ALS(userCol="user_id", itemCol="location_id", ratingCol="ratings", coldStartStrategy="drop")
        param_grid = ParamGridBuilder() \
            .addGrid(als.maxIter, [5, 10, 20]) \
            .addGrid(als.rank, [5, 10, 15]) \
            .addGrid(als.alpha, [0.8, 1.0, 1.2]) \
            .build()

        evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratings", predictionCol="prediction")
        pipeline = Pipeline(stages=[als])

        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=param_grid,
                                  evaluator=evaluator,
                                  numFolds=5)

        cv_model = crossval.fit(self.training)
        self.model = cv_model.bestModel

        best_params = {
            "maxIter": cv_model.bestModel.stages[-1]._java_obj.parent().getMaxIter(),
            "rank": cv_model.bestModel.stages[-1]._java_obj.parent().getRank(),
            "alpha": cv_model.bestModel.stages[-1]._java_obj.parent().getAlpha()
        }

        print("Optimized Parameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")

    def generate_recommendations(self, top_n=10):
        """Generate recommendations for all users and items."""
        user_recs = self.model.recommendForAllUsers(top_n)
        item_recs = self.model.recommendForAllItems(top_n)
        return user_recs, item_recs

    def save_model(self, path):
        """Save the trained model."""
        self.model.save(path)


