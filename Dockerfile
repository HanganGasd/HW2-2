# Use an official lightweight Python image.
FROM python:3.10-slim

WORKDIR /app

# Set environment variables for Python and Hugging Face optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

# Create a non-root user for security best practices
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
# (Provides build-essential for compiling some python packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies in a dedicated layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✨MLOps Optimization: Pre-download the Hugging Face model at build time.
# This bakes the model weights into the Docker image, preventing cold-start latency
# and network issues when the container first runs.
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis')"

# Copy the actual application code
COPY ./app ./app

# Change ownership of the application folder so the non-root user can read/write (e.g. cache)
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Start the application efficiently using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
