from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TwitterRoBERTaModel:
    def __init__(self):
        # Configurar o dispositivo (CPU ou GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar o tokenizer e o modelo pré-treinado
        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'cardiffnlp/twitter-roberta-base-sentiment'
        ).to(self.device)

        # Dicionário para mapear IDs de labels para categorias de sentimento
        self.id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    def analyze(self, text):
        # Pré-processar o texto (opcional, baseado nas recomendações do modelo)
        text = self.preprocess_text(text)

        # Tokenizar o texto
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

            # Calcular o score de sentimento entre -1 e 1
            class_values = torch.tensor([-1, 0, 1], dtype=torch.float).to(self.device)
            sentiment_score = torch.dot(probabilities, class_values).item()

            # Obter a categoria de sentimento com base na classe prevista
            predicted_class = torch.argmax(probabilities).item()
            sentiment_category = self.id2label[predicted_class]

        return sentiment_score, sentiment_category, None  # Retorna None para touchpoint

    def preprocess_text(self, text):
        # Função para pré-processar o texto conforme recomendado
        text = text.replace('@', '@user')
        # Adicione mais pré-processamentos se necessário
        return text
