from pydantic import BaseModel, Field

class SentimentRequest(BaseModel):
    text: str = Field(..., title="Text to analyze", description="The text input for sentiment analysis")

class SentimentResponse(BaseModel):
    label: str = Field(..., title="Sentiment Label", description="POSITIVE or NEGATIVE")
    score: float = Field(..., title="Confidence Score", description="Confidence score of the prediction")
