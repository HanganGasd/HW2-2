from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

class SentimentModel:
    def __init__(self):
        logger.info("Initializing the sentiment analysis model...")
        # Using HuggingFace's default lightweight pipeline for sentiment analysis (DistilBERT base).
        # Note: This is an English model by default. 
        # For Korean sentiment analysis, you can change the model to: model="monologg/koelectra-small-finetuned-nsmc"
        self.model = pipeline("sentiment-analysis", model="monologg/koelectra-small-finetuned-nsmc")
        logger.info("Model initialized successfully.")

    def predict(self, text: str) -> dict:
        # The pipeline returns a list of dictionaries, e.g., [{'label': 'POSITIVE', 'score': 0.99}]
        result = self.model(text)[0]
        label = str(result["label"]).upper()
        
        # Map labels to a uniform "POSITIVE" / "NEGATIVE" standard 
        if label in ["1", "LABEL_1", "POSITIVE"]:
            label = "POSITIVE"
        elif label in ["0", "LABEL_0", "NEGATIVE"]:
            label = "NEGATIVE"
            
        return {
            "label": label,
            "score": result["score"]
        }

# Global singleton instance to be loaded once upon server startup
sentiment_model = SentimentModel()
