from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity

class BERTModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
        self.touchpoint_descriptions = {
            "Search and Evaluate Professional Score": "User is searching for a doctor and evaluating the professional's score.",
            "Schedule Appointment": "User is scheduling an appointment with the doctor.",
            "Make Payment Online": "User is making an online payment for the consultation.",
            "Make Payment at Reception": "User is making a payment at the reception.",
            "Check-in Online": "User is checking in online before the consultation.",
            "Check-in at Reception": "User is checking in at the reception before the consultation.",
            "Access Platform for Online Consultation": "User is accessing the platform for an online consultation.",
            "Attend Online Consultation": "User is attending an online consultation with the doctor.",
            "Attend Offline Consultation": "User is attending an in-person consultation with the doctor.",
            "Follow-up Procedures": "User is following up with the doctor after an initial consultation.",
            "Leave Review and Feedback": "User is leaving a review or feedback about the consultation experience."
        }
        self.touchpoint_embeddings = self._compute_touchpoint_embeddings()

    def _compute_touchpoint_embeddings(self):
        embeddings = {}
        for touchpoint, description in self.touchpoint_descriptions.items():
            inputs = self.tokenizer(description, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model.distilbert(**inputs)  # Use 'distilbert' instead of 'bert'
            embeddings[touchpoint] = outputs.last_hidden_state.mean(dim=1).detach()
        return embeddings

    def analyze(self, text):
        # Análise de sentimento
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        sentiment_score = probabilities[0][2] - probabilities[0][0]
        sentiment_category = self.get_sentiment_category(sentiment_score.item())

        # Categorização de touchpoint
        touchpoint = self.suggest_touchpoint(text)

        return sentiment_score.item(), sentiment_category, touchpoint

    def get_sentiment_category(self, score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    def suggest_touchpoint(self, text):
        # Gerar embedding do texto de entrada
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model.distilbert(**inputs)  # Use 'distilbert' instead of 'bert'
        text_embedding = outputs.last_hidden_state.mean(dim=1).detach()

        # Calcular similaridade de cosseno entre o texto e os touchpoints
        similarities = {}
        for touchpoint, embedding in self.touchpoint_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(text_embedding, embedding)
            similarities[touchpoint] = similarity.item()

        suggested_touchpoint = max(similarities, key=similarities.get)
        return suggested_touchpoint

    def train(self, text, sentiment_score, touchpoint):
        print(f"Treinamento não implementado para BERT. Texto: {text}, sentimento: {sentiment_score}, touchpoint: {touchpoint}")
