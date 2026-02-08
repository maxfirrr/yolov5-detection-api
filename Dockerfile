FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p /app/models

# Copy application code
COPY main.py .

# Copy local models (if any)
COPY models/*.pt /app/models/ 2>/dev/null || true

# Environment variable for CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Volume for persistent model storage
VOLUME ["/app/models"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]