from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VADERModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text):
        scores = self.analyzer.polarity_scores(text)
        sentiment_score = scores['compound']
        
        if sentiment_score > 0.05:
            category = "Positive"
        elif sentiment_score < -0.05:
            category = "Negative"
        else:
            category = "Neutral"
        
        return sentiment_score, category

    def train(self, text, sentiment_score):
        # VADER is a rule-based model and doesn't require training
        print(f"VADER model doesn't require training. Text: {text}, sentiment: {sentiment_score}")
