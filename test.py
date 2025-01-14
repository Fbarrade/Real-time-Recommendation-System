# Import SentimentAnalyzer from utils folder
from utils.sentiment_analyzer import SentimentAnalyzer
import  pandas as pd
# Create an instance of the SentimentAnalyzer class
data = {
    "place_name": ["Place A", "Place A", "Place B", "Place B", "Place C", "Place A", "Place C"],
    "review": [
        "This place is amazing! Highly recommend it.",
        "Terrible service. I will never come back.",
        "The food was okay, but the ambiance was fantastic.",
        "I loved the experience. Will visit again soon!",
        "Worst experience ever. Avoid at all costs.",
        "Great service and excellent food.",
        "Not worth the price at all."
    ]
}

df = pd.DataFrame(data)

# Run Sentiment Analysis
sentiment_analyzer = SentimentAnalyzer(alpha=0.7)
processed_df, aggregated_scores = sentiment_analyzer.process_data(df)

# Display Results
print("Individual Reviews with Scores:")
print(processed_df)
print("\nCumulative Scores by Place:")
print(aggregated_scores)
