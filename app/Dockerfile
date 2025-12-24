# Use a slim Python image to keep the image small
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose ports (for documentation; Compose sets them)
EXPOSE 8000 8501