FROM python:3.9-slim

LABEL "com.github.actions.name"="SageMaker Training Action"
LABEL "com.github.actions.description"="Setup and trigger training jobs on Amazon SageMaker"
LABEL "com.github.actions.icon"="cloud"
LABEL "com.github.actions.color"="orange"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws/

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY schema/ ./schema/

# Copy entry point script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create non-root user for security
RUN useradd -m -u 1000 actionuser && chown -R actionuser:actionuser /app
USER actionuser

# Set entry point
ENTRYPOINT ["/entrypoint.sh"]