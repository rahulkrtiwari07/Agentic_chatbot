# Use slim Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for lancedb
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first (important)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose FastAPI port
EXPOSE 8002

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
