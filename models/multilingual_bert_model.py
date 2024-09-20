from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MultilingualBERTModel:
    def __init__(self):
        # Configurar o dispositivo (CPU ou GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar o tokenizer e o modelo pré-treinado
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'nlptown/bert-base-multilingual-uncased-sentiment'
        ).to(self.device)

        # Dicionário para mapear IDs de labels para categorias de sentimento
        self.id2label = {
            0: '1 star',
            1: '2 stars',
            2: '3 stars',
            3: '4 stars',
            4: '5 stars'
        }

    def analyze(self, text):
        # Pré-processar e tokenizar o texto
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        # Fazer a inferência
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Obter as probabilidades para cada classe
            probabilities = torch.softmax(logits, dim=-1)[0]

            # Mapear as classes para valores numéricos de -1 a 1
            class_values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).to(self.device)
            sentiment_score = torch.dot(probabilities, class_values).item()

            # Obter a categoria de sentimento com base na classe prevista
            predicted_class = torch.argmax(probabilities).item()
            sentiment_category = self.id2label[predicted_class]

        return sentiment_score, sentiment_category, None  # Retorna None para touchpoint
