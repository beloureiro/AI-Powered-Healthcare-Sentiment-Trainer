# the best so far *****
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from touchpoint_classifier import TouchpointClassifier  # Importa o classificador de touchpoints personalizado

# Classe que utiliza o modelo BERT multilíngue para análise de sentimento e touchpoints
class MultilingualBERTModel:
    def __init__(self):
        # Configura o dispositivo a ser usado (GPU se disponível, caso contrário CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carrega o tokenizer e o modelo pré-treinado 'bert-base-multilingual-uncased-sentiment'
        # Esse modelo é treinado para lidar com textos em vários idiomas e para fornecer classificações de sentimento em 5 estrelas
        self.tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'nlptown/bert-base-multilingual-uncased-sentiment'
        ).to(self.device)  # Move o modelo para o dispositivo configurado (CPU ou GPU)

        # Dicionário que mapeia os IDs de classes do modelo para categorias de estrelas (1 a 5)
        self.id2label = {
            0: '1 star',
            1: '2 stars',
            2: '3 stars',
            3: '4 stars',
            4: '5 stars'
        }

        # Inicializa o classificador de touchpoints para categorizar os pontos de interação com base no texto
        self.touchpoint_classifier = TouchpointClassifier()

    # Método principal que realiza a análise de sentimento e categorização de touchpoints
    def analyze(self, text):
        # Pré-processa e tokeniza o texto para ser compatível com o modelo BERT
        inputs = self.tokenizer(
            text,
            return_tensors='pt',  # Retorna tensores PyTorch
            truncation=True,       # Trunca o texto se for maior que o limite
            padding=True,          # Adiciona padding para manter o tamanho consistente
            max_length=512         # Limita o comprimento máximo do texto tokenizado
        ).to(self.device)  # Move o input para o dispositivo configurado (CPU ou GPU)

        # Realiza a inferência sem calcular gradientes (otimização de memória e processamento)
        with torch.no_grad():
            outputs = self.model(**inputs)  # Executa o modelo com o texto tokenizado
            logits = outputs.logits  # Obtém os logits, que são os valores brutos antes de serem convertidos em probabilidades

            # Aplica softmax para converter os logits em probabilidades para cada classe
            probabilities = torch.softmax(logits, dim=-1)[0]

            # Mapeia as classes de sentimento (1 a 5 estrelas) para um score numérico de -1.0 a 1.0
            class_values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]).to(self.device)
            sentiment_score = torch.dot(probabilities, class_values).item()

            # Obtém a classe prevista com a maior probabilidade
            predicted_class = torch.argmax(probabilities).item()
            raw_sentiment_category = self.id2label[predicted_class]  # Mapeia o ID para a categoria de estrela correspondente

            # Converte a categoria de estrelas para uma categoria de sentimento ('Positive', 'Neutral', 'Negative')
            sentiment_category = map_multilingual_category(raw_sentiment_category)

            # Classifica o touchpoint com base no texto analisado
            touchpoint = self.touchpoint_classifier.classify_touchpoint(text)

            return sentiment_score, sentiment_category, touchpoint

# Função auxiliar para mapear categorias de estrelas em sentimentos
def map_multilingual_category(category):
    if category in ['1 star', '2 stars']:
        return 'Negative'  # 1 ou 2 estrelas são consideradas sentimento negativo
    elif category == '3 stars':
        return 'Neutral'  # 3 estrelas é considerado sentimento neutro
    elif category in ['4 stars', '5 stars']:
        return 'Positive'  # 4 ou 5 estrelas são consideradas sentimento positivo
    else:
        return 'Neutral'  # Caso padrão, retorna sentimento neutro
