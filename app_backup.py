import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import time
import sys
import argparse
import logging
from typing import Tuple, Dict, Optional

from utils.landmark_utils import HandLandmarkExtractor, normalize_landmarks, PredictionSmoother
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignEvalApp:
    """Real-time ASL sign language evaluation application."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the application.

        Args:
            model_path: Path to a specific model to load (if None, load latest)
        """
        # Load model and metadata
        self.model, self.model_info = self._load_model(model_path)

        if self.model is None:
            raise ValueError("Failed to load model. Please train a model first.")

        # Initialize hand landmark extractor
        self.extractor = HandLandmarkExtractor(
            static_mode=False,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE
        )

        # Get class mapping
        self.idx_to_class = self.model_info['class_mapping']['idx_to_class']

        # Initialize prediction smoother
        self.smoother = PredictionSmoother(window_size=config.SMOOTHING_WINDOW_SIZE)

        logger.info("SignEval application initialized successfully.")

    def _load_model(self, model_path: Optional[str] = None) -> Tuple[Optional[tf.keras.Model], Optional[Dict]]:
        """
        Load the trained model and its metadata.

        Args:
            model_path: Path to a specific model file (if None, load latest)

        Returns:
            Tuple of (model, model_info)
        """
        if model_path is not None:
            # Load specific model
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None, None

            model = load_model(model_path)

            # Try to find model info
            model_dir = os.path.dirname(model_path)
            info_path = os.path.join(model_dir, 'model_info.pkl')

            if os.path.exists(info_path):
                with open(info_path, 'rb') as f:
                    model_info = pickle.load(f)
            else:
                logger.warning(f"Model info not found: {info_path}")
                # Create basic model info with default class mapping
                with open('data/processed/class_mapping.pkl', 'rb') as f:
                    model_info = {'class_mapping': pickle.load(f)}

            return model, model_info

        # Find latest model
        model_dirs = [d for d in os.listdir(config.MODEL_SAVE_DIR)
                     if os.path.isdir(os.path.join(config.MODEL_SAVE_DIR, d))]

        if not model_dirs:
            logger.error("No trained models found.")
            return None, None

        # Sort by timestamp
        latest_model_dir = sorted(model_dirs)[-1]
        model_path = os.path.join(config.MODEL_SAVE_DIR, latest_model_dir, 'best_model.h5')

        if not os.path.exists(model_path):
            model_path = os.path.join(config.MODEL_SAVE_DIR, latest_model_dir, 'final_model.h5')

        if not os.path.exists(model_path):
            logger.error(f"No model file found in {latest_model_dir}")
            return None, None

        # Load model
        model = load_model(model_path)

        # Load model info
        info_path = os.path.join(config.MODEL_SAVE_DIR, latest_model_dir, 'model_info.pkl')
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)

        logger.info(f"Loaded model from {model_path}")

        return model, model_info

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], Optional[float]]:
        """
        Process a video frame and make a prediction.

        Args:
            frame: Input video frame

        Returns:
            Tuple of (annotated_frame, prediction, confidence)
        """
        # Extract landmarks
        result = self.extractor.extract_landmarks(frame)

        if result is None:
            return frame, None, None

        landmarks, multi_hand_landmarks = result

        # Normalize landmarks
        normalized_landmarks = normalize_landmarks(landmarks)

        # Make prediction
        prediction = self.model.predict(np.array([normalized_landmarks]), verbose=0)[0]

        # Get predicted class and confidence
        pred_idx = np.argmax(prediction)
        confidence = prediction[pred_idx]
        pred_class = self.idx_to_class[pred_idx]

        # Apply smoothing
        smoothed_pred, smoothed_conf = self.smoother.update(pred_class, confidence)

        # Draw landmarks on the frame
        annotated_frame = self.extractor.draw_landmarks(frame, multi_hand_landmarks)

        # Only return prediction if confidence is above threshold
        if smoothed_conf >= config.PREDICTION_CONFIDENCE_THRESHOLD:
            return annotated_frame, smoothed_pred, smoothed_conf
        else:
            return annotated_frame, None, smoothed_conf

    def run(self, camera_index: int = 0, window_name: str = "SignEval - ASL Recognition"):
        """
        Run the application with webcam input.

        Args:
            camera_index: Index of the camera to use
            window_name: Name of the OpenCV window
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            logger.error(f"Failed to open camera at index {camera_index}")
            return

        # Set up window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # FPS calculation variables
        fps_counter = 0
        fps_start_time = time.time()
        fps = 0

        try:
            while True:
                # Read frame
                ret, frame = cap.read()

                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Process frame
                annotated_frame, prediction, confidence = self.process_frame(frame)

                # Calculate FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                # Add information to frame
                info_frame = annotated_frame.copy()

                # Add FPS
                cv2.putText(info_frame, f"FPS: {fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add hand detection status
                hand_status = "Hand Detected" if prediction is not None else "No Hand Detected"
                cv2.putText(info_frame, hand_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Add prediction
                if prediction is not None:
                    # Draw prediction with high confidence in green
                    cv2.putText(info_frame, f"Sign: {prediction} ({confidence:.2f})",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                elif confidence is not None:
                    # Draw low confidence prediction in yellow
                    cv2.putText(info_frame, f"Low confidence: {confidence:.2f}",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Add instructions
                cv2.putText(info_frame, "Press 'q' to quit", (10, info_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Display frame
                cv2.imshow(window_name, cv2.flip(info_frame, 1))  # Flip for mirror effect

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            self.extractor.close()
            logger.info("Application closed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SignEval - ASL Recognition")
    parser.add_argument('--model', type=str, help="Path to a specific model file", default=None)
    parser.add_argument('--camera', type=int, help="Camera index to use", default=0)

    return parser.parse_args()

def main():
    """Main function to run the application."""
    args = parse_args()

    try:
        app = SignEvalApp(model_path=args.model)
        app.run(camera_index=args.camera)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
