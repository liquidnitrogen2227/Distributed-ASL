from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import time
import os
import sys
import logging
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.landmark_utils import HandLandmarkExtractor, normalize_landmarks, PredictionSmoother
import config
from app import SignEvalApp

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Initialize the SignEval app
sign_app = SignEvalApp(model_path=os.environ.get('MODEL_PATH'))

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checking and load monitoring."""
    return jsonify({
        'status': 'healthy',
        'cpu': psutil.cpu_percent(),
        'memory': psutil.virtual_memory().percent,
        'timestamp': time.time()
    })

@app.route('/process', methods=['POST'])
def process_frame():
    """Process a frame and return the prediction."""
    start_time = time.time()
    
    # Get frame from request
    encoded_frame = request.json.get('frame')
    frame_data = base64.b64decode(encoded_frame)
    
    # Convert to OpenCV format
    np_arr = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Process frame
    annotated_frame, prediction, confidence = sign_app.process_frame(frame)
    
    # Encode processed frame
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_annotated = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(confidence) if confidence is not None else None,
        'annotated_frame': encoded_annotated,
        'processing_time': processing_time,
        'node_id': os.environ.get('NODE_ID', 'unknown')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)))