from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class RoBERTaModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_score = scores[0][2].item() - scores[0][0].item()  # Positive - Negative
        
        if sentiment_score > 0.1:
            category = "Positive"
        elif sentiment_score < -0.1:
            category = "Negative"
        else:
            category = "Neutral"
        
        return sentiment_score, category

    def train(self, text, sentiment_score):
        # In a real-world scenario, we would fine-tune the model here
        # For simplicity, we'll just print a message
        print(f"Training RoBERTa model with text: {text}, sentiment: {sentiment_score}")
