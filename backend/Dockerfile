# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including SSL tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    openssl \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads /etc/ssl/certs /etc/ssl/private

# Copy SSL setup script
COPY scripts/setup-ssl.sh /usr/local/bin/setup-ssl.sh
RUN chmod +x /usr/local/bin/setup-ssl.sh

# Expose ports (HTTP and HTTPS)
EXPOSE 8000 8443

# Copy startup script
COPY scripts/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Command to run the application
CMD ["/usr/local/bin/start.sh"]
