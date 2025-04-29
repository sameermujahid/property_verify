FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs \
    && mkdir -p /app/cache \
    && mkdir -p /app/uploads \
    && mkdir -p /app/model_cache \
    && mkdir -p /app/temp \
    && chown -R nobody:nogroup /app \
    && chmod -R 777 /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_md

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
ENV XDG_CACHE_HOME=/app/cache
ENV LOG_DIR=/app/logs

# Switch to non-root user
USER nobody

# Expose port
EXPOSE 7860

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"] 