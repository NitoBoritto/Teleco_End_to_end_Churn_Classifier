# Using a python docker hardened image
FROM python:3.11-slim

# Setting working directory inside the container
WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Explicitly ensure the model is where inference.py expects it
COPY src/serving/model /app/model

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Force the script to be an executable
RUN chmod +x start.sh

# Ensure linux line endings
RUN sed -i 's/\r$//' start.sh

# Run FastAPI app (pointing to your main.py where 'app = FastAPI()' is)
CMD ["./start.sh"]