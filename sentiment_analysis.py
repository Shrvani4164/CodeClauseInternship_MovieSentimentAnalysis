# Import required libraries
import pandas as pd  # For handling CSV files
import nltk  # For Natural Language Processing
from nltk.sentiment import SentimentIntensityAnalyzer  # For sentiment analysis

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
data = pd.read_csv('customer_reviews.csv')
def analyze_sentiment(text):
    if isinstance(text, str):  
        score = sia.polarity_scores(text)['compound']
        return 'Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral'
    else:
        return 'Unknown'  
data['Sentiment'] = data['review'].apply(analyze_sentiment)
data.to_csv('processed_reviews.csv', index=False)

print("Sentiment analysis complete! Results saved to 'processed_reviews.csv'.")
