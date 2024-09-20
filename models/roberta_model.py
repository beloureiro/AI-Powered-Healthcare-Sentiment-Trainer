from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Classe que encapsula o uso do modelo RoBERTa para classificação de sequências
class RoBERTaModel:
    def __init__(self):
        # Inicializa o tokenizer e o modelo pré-treinado do RoBERTa-base para classificação de sequência
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

        # Dicionário que mapeia touchpoints do ciclo de atendimento a suas descrições
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

        # Calcula os embeddings (representações vetoriais) das descrições dos touchpoints para usar na categorização
        self.touchpoint_embeddings = self._compute_touchpoint_embeddings()

    # Método privado para calcular os embeddings das descrições dos touchpoints
    def _compute_touchpoint_embeddings(self):
        embeddings = {}
        # Para cada touchpoint, tokeniza a descrição e gera o embedding usando o modelo RoBERTa
        for touchpoint, description in self.touchpoint_descriptions.items():
            inputs = self.tokenizer(description, return_tensors='pt', truncation=True, padding=True)
            # O modelo RoBERTa retorna o hidden state final para o input
            outputs = self.model.roberta(**inputs)
            # Calcula a média das representações escondidas para obter um vetor que represente a descrição
            embeddings[touchpoint] = outputs.last_hidden_state.mean(dim=1).detach()
        return embeddings

    # Método para analisar um texto, retornando o score de sentimento, a categoria de sentimento e o touchpoint sugerido
    def analyze(self, text):
        # Tokeniza o texto de entrada para análise de sentimento
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        # Passa o texto tokenizado pelo modelo para obter os logits (valores não normalizados)
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Converte os logits em probabilidades usando a função softmax
        probabilities = torch.softmax(logits, dim=1)
        # Calcula o score de sentimento (diferença entre as probabilidades de positivo e negativo)
        sentiment_score = probabilities[0][2] - probabilities[0][0]
        # Classifica o sentimento com base no score
        sentiment_category = self.get_sentiment_category(sentiment_score.item())

        # Sugere o touchpoint relacionado ao texto de entrada
        touchpoint = self.suggest_touchpoint(text)

        return sentiment_score.item(), sentiment_category, touchpoint

    # Método auxiliar para determinar a categoria de sentimento com base no score calculado
    def get_sentiment_category(self, score):
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Método para sugerir o touchpoint mais relevante com base no texto de entrada
    def suggest_touchpoint(self, text):
        # Tokeniza o texto de entrada e gera seu embedding
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model.roberta(**inputs)
        text_embedding = outputs.last_hidden_state.mean(dim=1).detach()

        # Calcula a similaridade de cosseno entre o embedding do texto e os embeddings dos touchpoints
        similarities = {}
        for touchpoint, embedding in self.touchpoint_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(text_embedding, embedding)
            similarities[touchpoint] = similarity.item()

        # Retorna o touchpoint com maior similaridade
        suggested_touchpoint = max(similarities, key=similarities.get)
        return suggested_touchpoint

    # Método de treinamento ainda não implementado, apenas retorna uma mensagem
    def train(self, *args, **kwargs):
        print("Treinamento não implementado para RoBERTa.")
