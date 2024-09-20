# models/bertweet_model.py
# este modelo nao esta sendo utilizado devido a necessidade de melhorar o processo de chunk, muitos erros ainda.limitado a textos longos maiores de 512 tokens

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from touchpoint_classifier import TouchpointClassifier
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from collections import Counter
import unicodedata

class BERTweetModel:
    def __init__(self):
        # Configure the device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the tokenizer and model from the same pre-trained model
        model_name = 'finiteautomata/bertweet-base-sentiment-analysis'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

        # Map label IDs to sentiment categories
        self.id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        # Initialize the Touchpoint Classifier
        self.touchpoint_classifier = TouchpointClassifier()

        # Maximum token limit
        self.max_tokens = 512

    def normalize_text(self, text):
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)
        # Replace special quotes and apostrophes with standard ones
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        return text

    def analyze(self, text):
        # Normalize the text
        text = self.normalize_text(text)

        # Tokenize and prepare the input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,  # Truncate sequences longer than max_length
            padding=True,
            max_length=self.max_tokens
        )

        # Move inputs to the device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Validate input_ids
        vocab_size = self.tokenizer.vocab_size
        if torch.any(inputs['input_ids'] >= vocab_size):
            invalid_tokens = inputs['input_ids'][inputs['input_ids'] >= vocab_size]
            print(f"Invalid token IDs found: {invalid_tokens}")
            raise ValueError("Invalid token IDs in input_ids.")

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            print(f"Logits shape: {logits.shape}")
            print(f"Logits: {logits}")

            # Get probabilities for each class
            probabilities = torch.softmax(logits, dim=-1)
            print(f"Probabilities shape: {probabilities.shape}")
            print(f"Probabilities: {probabilities}")

            # Confirm that probabilities and class_values have compatible shapes
            num_labels = self.model.config.num_labels
            class_values = torch.tensor([-1, 0, 1], dtype=torch.float).to(self.device)
            if len(class_values) != num_labels:
                raise ValueError(f"Incompatible number of class values: expected {num_labels}, got {len(class_values)}")

            # Calculate the sentiment score between -1 and 1
            sentiment_score = torch.dot(probabilities[0], class_values).item()

            # Get the sentiment category based on the predicted class
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            sentiment_category = self.id2label.get(predicted_class, "Unknown")

        # Use the touchpoint classifier on the original text
        touchpoint = self.touchpoint_classifier.classify_touchpoint(text)

        return sentiment_score, sentiment_category, touchpoint
