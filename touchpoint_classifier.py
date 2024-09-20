# touchpoint_classifier.py

from sentence_transformers import SentenceTransformer, util

class TouchpointClassifier:
    def __init__(self):
        # Initialize the Sentence Transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Define touchpoints and their descriptions
        self.touchpoints = {
            "Search and Evaluate Professional Score": "The patient uses the platform to search for a healthcare professional and evaluate their score.",
            "Schedule Appointment": "The patient schedules an appointment through the platform.",
            "Make Payment Online": "The patient makes the payment for the consultation online through the platform.",
            "Make Payment at Reception": "The patient makes the payment at the reception after the offline consultation.",
            "Check-in Online": "The patient completes an online check-in before the appointment.",
            "Check-in at Reception": "The patient checks in at the reception upon arriving at the clinic or hospital.",
            "Access Platform for Online Consultation": "The patient accesses the online platform to connect with the doctor for the consultation.",
            "Attend Online Consultation": "The patient attends the online consultation, where the doctor conducts the session through the platform.",
            "Attend Offline Consultation": "The patient attends an in-person consultation with the doctor at the clinic or hospital.",
            "Follow-up Procedures": "The patient follows up with any additional procedures, such as exams or surgeries, as prescribed by the doctor.",
            "Leave Review and Feedback": "The patient leaves a review and provides feedback about their overall experience."
        }

        # Compute embeddings for touchpoint descriptions
        self.touchpoint_embeddings = {}
        for touchpoint, description in self.touchpoints.items():
            embedding = self.model.encode(description, convert_to_tensor=True)
            self.touchpoint_embeddings[touchpoint] = embedding

    def classify_touchpoint(self, text):
        # Compute embedding for input text
        text_embedding = self.model.encode(text, convert_to_tensor=True)

        # Compute cosine similarity between text and each touchpoint
        similarities = {}
        for touchpoint, embedding in self.touchpoint_embeddings.items():
            similarity = util.pytorch_cos_sim(text_embedding, embedding).item()
            similarities[touchpoint] = similarity

        # Find touchpoint with highest similarity
        best_touchpoint = max(similarities, key=similarities.get)
        return best_touchpoint
