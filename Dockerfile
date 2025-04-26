FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
COPY dev-requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -r dev-requirements.txt

# Copy application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Make the run-game script executable
RUN chmod +x /usr/local/bin/run-game

# Switch to non-root user
USER appuser

# Run the game when the container starts
CMD ["run-game"]

# Health check to verify service is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "print('Health check passed')"
