FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask requests

# Copy application code
COPY . .

# Run the application
CMD ["python", "load_balancer/balancer.py"]