import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.nn import MSELoss
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Baixar os pacotes necessários do NLTK
nltk.download('punkt')
nltk.download('wordnet')

class BERTModel:
    def __init__(self, model_dir="HealthcareSentiment_BERT_v1", negative_words_file="data/cleaned_negative_words.csv"):
        self.model_dir = model_dir
        
        # Verificar e definir o dispositivo para GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Verificar se o modelo customizado já existe
        if os.path.exists(model_dir):
            print(f"Loading custom model from {model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        else:
            print("Loading default model and tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1).to(self.device)
        
        print(f"Model loaded on device: {self.device}")

        # Definir otimizador e função de perda
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.loss_fn_score = MSELoss()  # Função de perda para o grau de score (regressão)

        # Carregar palavras negativas do arquivo CSV
        self.negative_words = self.load_negative_words(negative_words_file)

        # Inicializar lematizador
        self.lemmatizer = WordNetLemmatizer()
        
        # Definir expressões críticas com pesos variáveis, sinônimos e construções gramaticais alternativas
        self.critical_negative_phrases = {
            'do not recommend': {'weight': -0.9, 'synonyms': ['would not recommend', 'wouldn\'t recommend', 'do not suggest', 'not worth it']},
            'worst': {'weight': -0.8, 'synonyms': ['absolute worst', 'most terrible', 'horrendous', 'dreadful', 'abysmal']},
            'poor service': {'weight': -0.7, 'synonyms': ['bad service', 'unprofessional service', 'lousy service', 'unacceptable service', 'terrible service']},
            'terrible': {'weight': -0.7, 'synonyms': ['horrible', 'awful', 'atrocious', 'abominable', 'dreadful']},
            'unprofessional': {'weight': -0.6, 'synonyms': ['incompetent', 'unskilled', 'inadequate', 'lacking professionalism']},
            'disrespectful': {'weight': -0.6, 'synonyms': ['rude', 'insulting', 'offensive', 'condescending', 'disdainful']},
            'never again': {'weight': -0.85, 'synonyms': ['will not return', 'no second chance', 'never coming back', 'no more visits']},
            'waste of time': {'weight': -0.75, 'synonyms': ['time wasted', 'complete waste of time', 'wasted my time', 'not worth my time']},
            'overpriced': {'weight': -0.6, 'synonyms': ['too expensive', 'rip-off', 'not worth the money', 'overcharging', 'unreasonably priced']},
            'no empathy': {'weight': -0.7, 'synonyms': ['lack of compassion', 'no compassion', 'unsympathetic', 'cold', 'uncaring']},
            'bad attitude': {'weight': -0.65, 'synonyms': ['negative attitude', 'rude behavior', 'hostile behavior', 'bad customer service']}
        }

        
        # Definir os touchpoints
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

    def load_negative_words(self, file_path):
        """
        Carregar as palavras negativas do arquivo CSV e armazená-las como um conjunto.
        """
        df = pd.read_csv(file_path)
        negative_words = set(df['term'].str.lower())  # Armazenar as palavras em minúsculas
        print(f"Loaded {len(negative_words)} negative words.")
        return negative_words

    def _compute_touchpoint_embeddings(self):
        embeddings = {}
        for touchpoint, description in self.touchpoint_descriptions.items():
            inputs = self.tokenizer(description, return_tensors='pt', truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_input_embeddings()(inputs['input_ids'])
                embedding = outputs.mean(dim=1).detach()
                embeddings[touchpoint] = embedding

            print(f"Embedding for touchpoint '{touchpoint}' computed. Shape: {embedding.shape}")

        if not embeddings:
            print("Error: Failed to compute embeddings for touchpoints.")
            raise ValueError("Failed to compute embeddings for touchpoints.")
        
        return embeddings

    def preprocess_text(self, text):
        """
        Função para realizar um pré-processamento leve no texto.
        Remove pontuações desnecessárias, normaliza a capitalização e faz lematização.
        """
        text = text.lower().strip()  # Normalizar a capitalização e remover espaços
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remover pontuação
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def analyze_multidimensional_sentiment(self, text):
        """
        Analisar o sentimento multidimensional, tratando cada sentença separadamente.
        """
        sentences = nltk.sent_tokenize(text)
        sentiment_scores = [self.analyze_sentence(sent) for sent in sentences]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        return avg_sentiment

    def analyze_sentence(self, sentence):
        # Analisar uma sentença individualmente
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        sentiment_score = torch.tanh(outputs.logits).item()
        return sentiment_score

    def adjust_sentiment_based_on_phrases(self, text, sentiment_score):
        """
        Ajusta o sentimento com base nas frases críticas e sinônimos.
        """
        # Iterar sobre todas as frases críticas e sinônimos
        for phrase, data in self.critical_negative_phrases.items():
            # Verificar se a frase ou um de seus sinônimos aparece no texto
            if any(synonym in text for synonym in [phrase] + data['synonyms']):
                sentiment_score += data['weight']
                print(f"Adjusting sentiment by {data['weight']} due to presence of phrase: {phrase}")

        # Garantir que o score se mantenha entre -1 e 1
        sentiment_score = max(min(sentiment_score, 1.0), -1.0)
        return sentiment_score

    def analyze(self, text):
        # Pré-processar o texto
        text = self.preprocess_text(text)

        # Verificar se o texto contém palavras ou expressões negativas
        negative_words_in_text = [word for word in text.split() if word in self.negative_words]
        critical_phrases_in_text = [phrase for phrase in self.critical_negative_phrases if phrase in text]

        # Analisar o texto com o BERT (multidimensional)
        sentiment_score = self.analyze_multidimensional_sentiment(text)

        # Ajustar o sentimento com base nas palavras negativas e expressões críticas
        sentiment_score = self.adjust_sentiment_based_on_phrases(text, sentiment_score)

        # Mapeamento para categoria (Negativo, Neutro, Positivo)
        sentiment_category = self.get_sentiment_category(sentiment_score)

        # Categorização de touchpoint
        touchpoint = self.suggest_touchpoint(text)

        return sentiment_score, sentiment_category, touchpoint

    def get_sentiment_category(self, score):
        """
        Mapeamento do score contínuo para as categorias de sentimento.
        - Negativo: score < -0.01
        - Neutro: -0.01 <= score <= 0.01
        - Positivo: score > 0.01
        """
        if score > 0.01:
            return "Positive"
        elif score < -0.01:
            return "Negative"
        else:
            return "Neutral"

    def suggest_touchpoint(self, text):
        """
        Sugere um touchpoint com base na similaridade do texto com os embeddings de touchpoints.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_input_embeddings()(inputs['input_ids'])
            text_embedding = outputs.mean(dim=1).detach()

        if text_embedding is None or text_embedding.shape[0] == 0:
            print("Error: Failed to generate embeddings for the input text.")
            raise ValueError("Failed to generate embeddings for the input text.")
        
        similarities = {}
        for touchpoint, embedding in self.touchpoint_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(text_embedding, embedding)
            similarities[touchpoint] = similarity.item()

        suggested_touchpoint = max(similarities, key=similarities.get)
        print(f"Suggested touchpoint: {suggested_touchpoint}")
        return suggested_touchpoint

    def train(self, text, score_value, touchpoint):
        """
        Função para treinar o modelo BERT usando o texto fornecido.
        """
        print(f"Training model. Text: {text}, score: {score_value}, touchpoint: {touchpoint}")

        # Preparar os dados de entrada e rótulos para o treinamento
        self.model.train()

        # Tokenizar o texto e mover para a GPU
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        
        # Fazer uma passada para frente e calcular a perda
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Rótulos de score (grau entre -1 e 1) - convertido para tensor
        target_score = torch.tensor([score_value], dtype=torch.float).to(self.device)

        # Calcular a perda
        loss_score = self.loss_fn_score(torch.tanh(logits), target_score)

        # Backpropagation e atualização dos pesos
        self.optimizer.zero_grad()
        loss_score.backward()
        self.optimizer.step()

        print(f"Training complete. Total Loss: {loss_score.item()}")

    def save_model(self):
        """
        Função para salvar o modelo e o tokenizer.
        """
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        print(f"Custom model saved to {self.model_dir}")

