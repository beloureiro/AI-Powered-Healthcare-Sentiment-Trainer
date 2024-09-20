from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Classe que usa o modelo VADER para análise de sentimentos e palavras-chave para categorização de touchpoints
class VADERModel:
    def __init__(self):
        # Inicializa o analisador de sentimentos VADER
        self.analyzer = SentimentIntensityAnalyzer()

        # Dicionário que contém padrões de palavras-chave para identificar os touchpoints
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

    # Função principal que realiza a análise de sentimento e sugere um touchpoint com base no texto
    def analyze(self, text):
        # Análise de sentimento usando o VADER
        vs = self.analyzer.polarity_scores(text)  # Obtém as pontuações de sentimento do texto
        sentiment_score = vs['compound']  # A pontuação 'compound' é um score geral de sentimento
        sentiment_category = self.get_sentiment_category(sentiment_score)  # Classifica o sentimento como Positivo, Negativo ou Neutro

        # Categorização de touchpoint com base nas palavras-chave do texto
        touchpoint = self.suggest_touchpoint(text)

        return sentiment_score, sentiment_category, touchpoint

    # Método para determinar a categoria de sentimento com base no score do VADER
    def get_sentiment_category(self, score):
        if score >= 0.05:
            return "Positive"  # Sentimento positivo
        elif score <= -0.05:
            return "Negative"  # Sentimento negativo
        else:
            return "Neutral"  # Sentimento neutro

    # Método para sugerir o touchpoint com base nas palavras-chave presentes no texto
    def suggest_touchpoint(self, text):
        text = text.lower()  # Converte o texto para minúsculas para garantir que a busca de palavras seja case-insensitive
        # Verifica se o texto corresponde a algum dos padrões de palavras-chave nos touchpoints
        for pattern, touchpoint in self.touchpoint_keywords.items():
            if re.search(pattern, text):  # Se encontrar uma correspondência, retorna o touchpoint
                return touchpoint
        # Se nenhuma palavra-chave for encontrada, retorna um touchpoint padrão
        return "Search and Evaluate Professional Score"

    # Método de treinamento que imprime uma mensagem. O VADER não requer treinamento.
    def train(self, *args, **kwargs):
        print("VADER não requer treinamento.")
