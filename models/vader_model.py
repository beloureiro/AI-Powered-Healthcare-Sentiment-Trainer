from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

class VADERModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.touchpoint_keywords = {
            r"\b(search|evaluate|score|rating|professional|doctor review)\b": "Search and Evaluate Professional Score",
            r"\b(schedule|appointment|booking|availability|reschedule)\b": "Schedule Appointment",
            r"\b(payment|pay|billing|cost|charge|online payment)\b": "Make Payment Online",
            r"\b(reception|payment at reception)\b": "Make Payment at Reception",
            r"\b(check-in|check in|arrival|online check-in)\b": "Check-in Online",
            r"\b(check-in|check in|reception)\b": "Check-in at Reception",
            r"\b(platform|online consultation|virtual consultation)\b": "Access Platform for Online Consultation",
            r"\b(consultation|online visit|telemedicine)\b": "Attend Online Consultation",
            r"\b(offline consultation|in-person visit|doctor)\b": "Attend Offline Consultation",
            r"\b(follow-up|exam|surgery|procedure)\b": "Follow-up Procedures",
            r"\b(review|feedback|rate|experience|opinion)\b": "Leave Review and Feedback"
        }

    def analyze(self, text):
        # Análise de sentimento
        vs = self.analyzer.polarity_scores(text)
        sentiment_score = vs['compound']
        sentiment_category = self.get_sentiment_category(sentiment_score)

        # Categorização de touchpoint
        touchpoint = self.suggest_touchpoint(text)

        return sentiment_score, sentiment_category, touchpoint

    def get_sentiment_category(self, score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def suggest_touchpoint(self, text):
        text = text.lower()
        for pattern, touchpoint in self.touchpoint_keywords.items():
            if re.search(pattern, text):
                return touchpoint
        # Retorna um touchpoint padrão se não houver correspondência
        return "Search and Evaluate Professional Score"

    def train(self, text, sentiment_score, touchpoint):
        print(f"VADER não requer treinamento. Texto: {text}, sentimento: {sentiment_score}, touchpoint: {touchpoint}")
