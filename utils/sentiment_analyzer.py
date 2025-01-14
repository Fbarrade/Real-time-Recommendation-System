import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, alpha=0.7):
        self.alpha = alpha  # Weight for VADER scores
        self.analyzer = SentimentIntensityAnalyzer()
        self.transformer_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    def rule_based_sentiment_analysis(self, reviews):
        results = []
        for text in reviews:
            vader_score = self.analyzer.polarity_scores(text)['compound']
            results.append({"vader_score": vader_score})
        return results
    
    def transformer_sentiment_analysis(self, reviews):
        reviews_list = reviews.tolist() if isinstance(reviews, pd.Series) else reviews
        results = self.transformer_analyzer(reviews_list)
        return [{"transformer_score": res['label']} for res in results]
    
    def normalize_vader(self, score):
        return 2 * (score + 1)  # Normalizes VADER score from [-1, 1] to [1, 5]
    
    def weighted_combination(self, vader_score, transformer_score):
        return self.alpha * self.normalize_vader(vader_score) + (1 - self.alpha) * transformer_score
    
    def calculate_cumulative_scores(self, group):
        group["normalized_vader"] = group["vader_score"].apply(self.normalize_vader)
        group["numeric_transformer"] = group["transformer_score"].apply(lambda x: int(x[0]))
        weighted_score = group.apply(
            lambda row: self.weighted_combination(row["vader_score"], row["numeric_transformer"]), axis=1
        )
        return pd.Series({
            "vader_cumulative": group["vader_score"].mean(),
            "transformer_cumulative": group["numeric_transformer"].mean(),
            "weighted_cumulative": weighted_score.mean()
        })
    
    def process_data(self, df):
        rule_based_results = self.rule_based_sentiment_analysis(df["review"])
        transformer_results = self.transformer_sentiment_analysis(df["review"])
        
        df["vader_score"] = [res["vader_score"] for res in rule_based_results]
        df["transformer_score"] = [res["transformer_score"] for res in transformer_results]
        
        aggregated_scores = df.groupby("place_name").apply(self.calculate_cumulative_scores).reset_index()
        return df, aggregated_scores



