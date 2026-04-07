import logging
from fastapi import FastAPI
from .api import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="MLOps Sentiment Analysis API",
    description="A simple API for text sentiment analysis serving as an MLOps pipeline foundation.",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    # Entry point for running the application directly
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
