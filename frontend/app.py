from flask import Flask, render_template, Response, jsonify
import cv2
import time
import base64
import numpy as np
import requests
import json
import logging
import os
from typing import Dict, Optional, Tuple

app = Flask(__name__, static_folder='static', template_folder='templates')
logger = logging.getLogger(__name__)

# Configuration
LOAD_BALANCER_URL = os.environ.get('LOAD_BALANCER_URL', 'http://load_balancer:5000')

class FrontendApp:
    def __init__(self):
        self.cap = None
        self.camera_index = 0
        self.is_streaming = False
        self.last_frame = None
        self.last_prediction = None
        self.last_confidence = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        self.processing_latency = 0
        
    def start_capture(self, camera_index=0):
        """Start video capture."""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera at index {camera_index}")
            return False
            
        self.is_streaming = True
        return True
    
    def stop_capture(self):
        """Stop video capture."""
        self.is_streaming = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def process_frame(self) -> Tuple[Optional[np.ndarray], Dict]:
        """Process the current frame and return results."""
        if not self.is_streaming or self.cap is None:
            return None, {}
            
        ret, frame = self.cap.read()
        
        if not ret:
            logger.error("Failed to read frame from camera")
            return None, {}
            
        # Calculate FPS
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
            
        # Send frame for processing
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{LOAD_BALANCER_URL}/process",
                json={'frame': encoded_frame},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Decode processed frame
                annotated_frame_data = base64.b64decode(result['annotated_frame'])
                np_arr = np.frombuffer(annotated_frame_data, dtype=np.uint8)
                annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # Mirror for display
                mirrored_frame = cv2.flip(annotated_frame, 1)
                
                # Update metrics
                self.last_prediction = result.get('prediction')
                self.last_confidence = result.get('confidence')
                self.processing_latency = time.time() - start_time
                
                # Add load balancing info
                load_balancer_info = result.get('load_balancer', {})
                node_id = load_balancer_info.get('node_id', 'unknown')
                algorithm = load_balancer_info.get('algorithm', 'unknown')
                
                # Create metrics dict
                metrics = {
                    'fps': self.fps,
                    'latency': self.processing_latency,
                    'prediction': self.last_prediction,
                    'confidence': self.last_confidence,
                    'node_id': node_id,
                    'algorithm': algorithm
                }
                
                return mirrored_frame, metrics
                
            else:
                logger.error(f"Load balancer error: {response.status_code}")
                return None, {'error': f"Load balancer error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None, {'error': str(e)}

# Frontend app instance
frontend = FrontendApp()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    """Generate video frames for streaming."""
    while frontend.is_streaming:
        frame, metrics = frontend.process_frame()
        
        if frame is None:
            continue
            
        # Add text with metrics
        h, w = frame.shape[:2]
        info_layer = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add FPS
        cv2.putText(info_layer, f"FPS: {metrics['fps']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # Add latency
        cv2.putText(info_layer, f"Latency: {metrics['latency']:.2f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # Add node info
        cv2.putText(info_layer, f"Node: {metrics.get('node_id', 'unknown')}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        # Add prediction
        if metrics.get('prediction') is not None:
            cv2.putText(info_layer, f"Sign: {metrics['prediction']} ({metrics['confidence']:.2f})",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        # Overlay info layer onto frame
        mask = info_layer > 0
        frame[mask] = info_layer[mask]
        
        # Convert to JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_stream():
    """Start video streaming."""
    success = frontend.start_capture(camera_index=0)
    if success:
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to open camera'}), 500

@app.route('/stop', methods=['POST'])
def stop_stream():
    """Stop video streaming."""
    frontend.stop_capture()
    return jsonify({'status': 'success'})

@app.route('/metrics')
def get_metrics():
    """Get current metrics from load balancer."""
    try:
        response = requests.get(f"{LOAD_BALANCER_URL}/metrics")
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f"Failed to get metrics: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/algorithm/<algorithm>', methods=['POST'])
def set_algorithm(algorithm):
    """Change load balancing algorithm."""
    try:
        response = requests.post(f"{LOAD_BALANCER_URL}/algorithm/{algorithm}")
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f"Failed to set algorithm: {response.status_code}"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)