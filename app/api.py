from fastapi import APIRouter, HTTPException
from .schemas import SentimentRequest, SentimentResponse
from .model import sentiment_model
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    logger.info(f"Received text for prediction: {request.text[:50]}...")
    try:
        prediction = sentiment_model.predict(request.text)
        return SentimentResponse(
            label=prediction["label"],
            score=prediction["score"]
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@router.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": True}
