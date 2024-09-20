import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.nn import CrossEntropyLoss
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import unicodedata

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

class BERTModel:
    def __init__(self, model_dir="HealthcareSentiment_BERT_v1", negative_words_file="data/cleaned_negative_words.csv"):
        self.model_dir = model_dir
        
        # Verify and set device to GPU only
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            raise RuntimeError("GPU is not available. Please ensure that a GPU is available and PyTorch is installed with CUDA support.")
        
        # Check if custom model exists
        if os.path.exists(model_dir):
            print(f"Loading custom model from {model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        else:
            print("Loading pre-trained fine-tuned model for sentiment analysis.")
            model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        
        print(f"Model loaded on device: {self.device}")

        # Define optimizer and loss function (CrossEntropyLoss for classification)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.loss_fn = CrossEntropyLoss()

        # Load negative words from CSV file
        self.negative_words = self.load_negative_words(negative_words_file)

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Define critical negative phrases with weights and synonyms
        self.critical_negative_phrases = {
            # ... (your existing phrases)
        }

        # Define touchpoints
        self.touchpoint_descriptions = {
            # ... (your existing touchpoints)
        }
        self.touchpoint_embeddings = self._compute_touchpoint_embeddings()

    def load_negative_words(self, file_path):
        """
        Load negative words from CSV and store them as a set.
        """
        df = pd.read_csv(file_path)
        negative_words = set(df['term'].str.lower())  # Store words in lowercase
        print(f"Loaded {len(negative_words)} negative words.")
        return negative_words

    def _compute_touchpoint_embeddings(self):
        embeddings = {}
        for touchpoint, description in self.touchpoint_descriptions.items():
            inputs = self.tokenizer(
                description,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            with torch.no_grad():
                # Access the base model (DistilBERT) to get embeddings
                outputs = self.model.distilbert(**inputs)
                last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]
                embedding = last_hidden_state.mean(dim=1).detach()  # Mean pooling over sequence length
                embeddings[touchpoint] = embedding

            print(f"Embedding for touchpoint '{touchpoint}' computed. Shape: {embedding.shape}")

        if not embeddings:
            print("Error: Failed to compute embeddings for touchpoints.")
            raise ValueError("Failed to compute embeddings for touchpoints.")

        return embeddings

    def preprocess_text(self, text):
        """
        Light preprocessing of the text.
        Removes unnecessary punctuation, normalizes capitalization, and lemmatizes.
        """
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        # Replace special quotes with standard ones
        text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
        text = text.lower().strip()  # Normalize capitalization and remove spaces
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove punctuation
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def analyze_multidimensional_sentiment(self, text):
        """
        Analyze multidimensional sentiment by treating each sentence separately.
        """
        sentences = nltk.sent_tokenize(text)
        sentiment_scores = [self.analyze_sentence(sent) for sent in sentences]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        return avg_sentiment

    def analyze_sentence(self, sentence):
        # Analyze a single sentence
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [1, 2]
            probabilities = torch.softmax(logits, dim=-1)[0]  # Shape: [2]

            # Map probabilities to sentiment score between -1 and 1
            class_values = torch.tensor([-1.0, 1.0]).to(self.device)
            sentiment_score = torch.dot(probabilities, class_values).item()
        return sentiment_score

    def adjust_sentiment_based_on_phrases(self, text, sentiment_score):
        """
        Adjusts the sentiment based on critical phrases and synonyms.
        """
        # Iterate over all critical phrases and synonyms
        for phrase, data in self.critical_negative_phrases.items():
            # Check if the phrase or any of its synonyms appear in the text
            if any(synonym in text for synonym in [phrase] + data['synonyms']):
                sentiment_score += data['weight']
                print(f"Adjusting sentiment by {data['weight']} due to presence of phrase: {phrase}")

        # Ensure the score stays between -1 and 1
        sentiment_score = max(min(sentiment_score, 1.0), -1.0)
        return sentiment_score

    def analyze(self, text):
        # Preprocess the text
        text = self.preprocess_text(text)

        # Check for negative words and critical phrases
        negative_words_in_text = [word for word in text.split() if word in self.negative_words]
        critical_phrases_in_text = [phrase for phrase in self.critical_negative_phrases if phrase in text]

        # Analyze the text with BERT (multidimensional)
        sentiment_score = self.analyze_multidimensional_sentiment(text)

        # Adjust sentiment based on negative words and critical phrases
        sentiment_score = self.adjust_sentiment_based_on_phrases(text, sentiment_score)

        # Map to sentiment category (Negative, Neutral, Positive)
        sentiment_category = self.get_sentiment_category(sentiment_score)

        # Suggest touchpoint
        touchpoint = self.suggest_touchpoint(text)

        return sentiment_score, sentiment_category, touchpoint

    def get_sentiment_category(self, score):
        """
        Map the continuous score to sentiment categories.
        - Negative: score < -0.01
        - Neutral: -0.01 <= score <= 0.01
        - Positive: score > 0.01
        """
        if score > 0.01:
            return "Positive"
        elif score < -0.01:
            return "Negative"
        else:
            return "Neutral"

    def suggest_touchpoint(self, text):
        """
        Suggests a touchpoint based on the similarity of the text to touchpoint embeddings.
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        with torch.no_grad():
            # Access the base model to get embeddings
            outputs = self.model.distilbert(**inputs)
            last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]
            text_embedding = last_hidden_state.mean(dim=1).detach()

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
        Function to fine-tune the BERT model using the provided text.
        """
        print(f"Training model. Text: {text}, score: {score_value}, touchpoint: {touchpoint}")

        # Prepare the data
        self.model.train()

        # Tokenize text and move to GPU
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Map score_value to classification label (0: Negative, 1: Positive)
        label = 1 if score_value > 0 else 0
        labels = torch.tensor([label], dtype=torch.long).to(self.device)  # CrossEntropyLoss expects labels as long

        # Forward pass and compute loss
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss  # CrossEntropyLoss computed internally

        # Backpropagation and weight update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f"Training complete. Total Loss: {loss.item()}")

    def save_model(self):
        """
        Function to save the model and tokenizer.
        """
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        print(f"Custom model saved to {self.model_dir}")
