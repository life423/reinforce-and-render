# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Environment variables for configuration
ENV PYTHONUNBUFFERED=1

# Expose the port used by your game (if needed, e.g., if you open a web interface)
EXPOSE 8000

# Command to run your game
CMD ["python", "main.py"]
