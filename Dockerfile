# Use a slim Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed to build packages like noise
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Set environment variables (e.g., to load MongoDB credentials from .env)
ENV PYTHONUNBUFFERED=1

# Run the main Python file when the container starts
CMD ["python", "main.py"]
