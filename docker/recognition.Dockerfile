FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir flask numpy opencv-python-headless tensorflow mediapipe psutil requests

# Copy application code
COPY . .

# Run the application
CMD ["python", "recognition/service.py"]