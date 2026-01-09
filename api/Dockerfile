# Use Python 3.11 for TensorFlow compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY api/requirements.txt /app/api/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/api/requirements.txt

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/models /app/data/train /app/data/test /app/ui

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Run the application
# Use the PORT environment variable provided by Railway
CMD cd /app/api && uvicorn main:app --host 0.0.0.0 --port $PORT
