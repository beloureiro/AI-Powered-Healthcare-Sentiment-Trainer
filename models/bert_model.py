import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.nn import CrossEntropyLoss, MSELoss

class BERTModel:
    def __init__(self, model_dir="HealthcareSentiment_BERT_v1"):
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
            self.model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(self.device)
        
        print(f"Model loaded on device: {self.device}")

        # Definir otimizador e função de perda
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.loss_fn_category = CrossEntropyLoss()  # Função de perda para a categoria de sentimento
        self.loss_fn_score = MSELoss()  # Função de perda para o grau de score (regressão)

        self.touchpoint_descriptions = {
            # Touchpoints...
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
            inputs = self.tokenizer(description, return_tensors='pt', truncation=True, padding=True).to(self.device)
            with torch.no_grad():  # Evitar o cálculo do gradiente durante a geração de embeddings
                outputs = self.model.get_input_embeddings()(inputs['input_ids'])  # Corrigido para usar a camada de embeddings corretamente
                embedding = outputs.mean(dim=1).detach()  # Obtendo a média dos embeddings gerados
                embeddings[touchpoint] = embedding

            print(f"Embedding for touchpoint '{touchpoint}' computed. Shape: {embedding.shape}")  # Log para cada touchpoint

        if not embeddings:
            print("Error: Failed to compute embeddings for touchpoints.")
            raise ValueError("Failed to compute embeddings for touchpoints.")
        
        return embeddings

    def analyze(self, text):
        # Analisar o texto
        print(f"Analyzing on device: {self.device}")
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
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
        # Gerar embedding do texto de entrada e mover para a GPU
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_input_embeddings()(inputs['input_ids'])
            text_embedding = outputs.mean(dim=1).detach()

        # Verificar se os embeddings foram gerados corretamente
        if text_embedding is None or text_embedding.shape[0] == 0:
            print("Error: Failed to generate embeddings for the input text.")
            raise ValueError("Failed to generate embeddings for the input text.")
        
        print(f"Text embedding shape: {text_embedding.shape}")  # Log para verificar a forma do embedding do texto

        # Calcular similaridade de cosseno entre o texto e os touchpoints
        similarities = {}
        for touchpoint, embedding in self.touchpoint_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(text_embedding, embedding)
            similarities[touchpoint] = similarity.item()
        
        # Log das similaridades calculadas
        print(f"Similarities calculated: {similarities}")

        # Verificar se o dicionário de similaridades foi preenchido
        if not similarities:
            print("Error: No similarities were calculated between the text and touchpoints.")
            raise ValueError("No similarities were calculated between the text and touchpoints.")
        
        suggested_touchpoint = max(similarities, key=similarities.get)
        print(f"Suggested touchpoint: {suggested_touchpoint}")  # Log do touchpoint sugerido
        return suggested_touchpoint


    def train(self, text, sentiment_category, score_value, touchpoint):
        print(f"Training model. Text: {text}, sentiment category: {sentiment_category}, score: {score_value}, touchpoint: {touchpoint}")
        
        # Mapear as categorias de sentimento para valores numéricos
        sentiment_mapping = {
            "Positive": 2,
            "Neutral": 1,
            "Negative": 0
        }
        
        if sentiment_category not in sentiment_mapping:
            raise ValueError(f"Sentiment category '{sentiment_category}' is not recognized.")
        
        # Preparar os dados de entrada e rótulos para o treinamento
        self.model.train()

        # Tokenizar o texto e mover para a GPU
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        
        # Fazer uma passada para frente e calcular a perda
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Rótulos de categoria (Negativo, Neutro, Positivo) - convertido para tensor
        target_category = torch.tensor([sentiment_mapping[sentiment_category]], dtype=torch.long).to(self.device)
        
        # Rótulos de score (grau entre -1 e 1) - também convertido para tensor
        target_score = torch.tensor([score_value], dtype=torch.float).to(self.device)

        # Calcular a perda
        loss_category = self.loss_fn_category(logits, target_category)
        loss_score = self.loss_fn_score(torch.softmax(logits, dim=1), target_score)

        # Perda total
        total_loss = loss_category + loss_score

        # Backpropagation e atualização dos pesos
        self.optimizer.zero_grad()  # Zerar os gradientes
        total_loss.backward()  # Retropropagar
        self.optimizer.step()  # Atualizar os pesos

        print(f"Training complete. Total Loss: {total_loss.item()}")


    def save_model(self):
        # Salvar o modelo e tokenizer
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        print(f"Custom model saved to {self.model_dir}")
