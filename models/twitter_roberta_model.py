from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from touchpoint_classifier import TouchpointClassifier  # Importa o classificador de touchpoints personalizado

# Classe que utiliza o modelo RoBERTa específico para análise de sentimentos no Twitter
class TwitterRoBERTaModel:
    def __init__(self):
        # Configura o dispositivo a ser usado (GPU se disponível, caso contrário CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carrega o tokenizer e o modelo pré-treinado 'twitter-roberta-base-sentiment' para análise de sentimentos em textos do Twitter
        self.tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'cardiffnlp/twitter-roberta-base-sentiment'
        ).to(self.device)  # Move o modelo para o dispositivo configurado (CPU ou GPU)

        # Mapeia os IDs das classes retornadas pelo modelo para categorias de sentimento (Negativo, Neutro, Positivo)
        self.id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        # Inicializa o classificador de touchpoints, responsável por classificar o estágio do processo com base no texto
        self.touchpoint_classifier = TouchpointClassifier()

    def analyze(self, text):
        # Pré-processa o texto, substituindo menções (@) para um formato genérico
        text = self.preprocess_text(text)

        # Tokeniza o texto para que possa ser processado pelo modelo (adiciona truncamento e padding se necessário)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',  # Retorna tensores PyTorch
            truncation=True,       # Trunca o texto se ultrapassar o limite
            padding=True,          # Adiciona padding para manter o comprimento consistente
            max_length=512         # Limita o comprimento máximo do texto tokenizado
        ).to(self.device)  # Move o input para o dispositivo configurado (CPU ou GPU)

        # Realiza a inferência no modelo sem calcular gradientes (economia de memória e tempo)
        with torch.no_grad():
            outputs = self.model(**inputs)  # Realiza a inferência para o texto
            logits = outputs.logits  # Obtém os logits, que são os valores brutos antes da softmax

            # Aplica a função softmax para converter os logits em probabilidades
            probabilities = torch.softmax(logits, dim=-1)[0]

            # Calcula um score de sentimento contínuo entre -1 (negativo) e 1 (positivo)
            class_values = torch.tensor([-1, 0, 1], dtype=torch.float).to(self.device)
            sentiment_score = torch.dot(probabilities, class_values).item()

            # Obtém a classe de sentimento prevista (a classe com a maior probabilidade)
            predicted_class = torch.argmax(probabilities).item()
            sentiment_category = self.id2label[predicted_class]  # Converte o ID da classe para a categoria correspondente

        # Classifica o touchpoint com base no texto analisado
        touchpoint = self.touchpoint_classifier.classify_touchpoint(text)

        return sentiment_score, sentiment_category, touchpoint

    def preprocess_text(self, text):
        # Substitui menções no texto (ex: '@usuario') por '@user', conforme recomendado para este modelo
        text = text.replace('@', '@user')
        # Adicione mais pré-processamento, se necessário, como remoção de URLs ou emojis
        return text
