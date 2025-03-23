import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandLandmarkExtractor:
    def __init__(self, static_mode: bool = True, max_hands: int = 1, min_detection_confidence: float = 0.7):
        """
        Initialize the hand landmark extractor.

        Args:
            static_mode: Whether to treat input as static images
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
        """
        self.hands = mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence
        )

    def extract_landmarks(self, image: np.ndarray) -> Optional[Tuple[List[float], List]]:
        """
        Extract hand landmarks from an image.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (landmark_list, multi_hand_landmarks) if hand detected, None otherwise
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        # Get landmarks from the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []

        # Extract x, y, z coordinates for each landmark
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])

        return landmarks, results.multi_hand_landmarks

    def draw_landmarks(self, image: np.ndarray, multi_hand_landmarks: List) -> np.ndarray:
        """
        Draw hand landmarks on an image.

        Args:
            image: Input image
            multi_hand_landmarks: List of hand landmarks from MediaPipe

        Returns:
            Image with landmarks drawn
        """
        annotated_image = image.copy()
        for hand_landmarks in multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        return annotated_image

    def close(self):
        """Close the hands object to release resources."""
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def normalize_landmarks(landmarks: List[float]) -> List[float]:
    """
    Normalize landmarks to make them invariant to scale and translation.

    Args:
        landmarks: List of landmarks [x1, y1, z1, x2, y2, z2, ...]

    Returns:
        Normalized landmarks
    """
    # Reshape to (21, 3) for easier processing
    landmarks_array = np.array(landmarks).reshape(-1, 3)

    # Use wrist (landmark 0) as origin
    wrist = landmarks_array[0]
    landmarks_array = landmarks_array - wrist

    # Scale by the distance from wrist to middle finger MCP (landmark 9)
    scale_reference = np.linalg.norm(landmarks_array[9])
    if scale_reference > 0:
        landmarks_array = landmarks_array / scale_reference

    # Flatten back to a list
    return landmarks_array.flatten().tolist()


class PredictionSmoother:
    def __init__(self, window_size: int = 5):
        """
        Initialize prediction smoother for more stable real-time predictions.

        Args:
            window_size: Number of frames to consider for smoothing
        """
        self.window_size = window_size
        self.predictions = []

    def update(self, prediction: str, confidence: float) -> Tuple[str, float]:
        """
        Update prediction history and get smoothed prediction.

        Args:
            prediction: Current prediction
            confidence: Confidence of the prediction

        Returns:
            Most frequent prediction in the window and its average confidence
        """
        self.predictions.append((prediction, confidence))

        # Keep only the last window_size predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)

        # Count occurrences of each prediction
        prediction_counts = {}
        prediction_confidences = {}

        for pred, conf in self.predictions:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                prediction_confidences[pred] = []

            prediction_counts[pred] += 1
            prediction_confidences[pred].append(conf)

        # Find the most frequent prediction
        most_frequent = max(prediction_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = sum(prediction_confidences[most_frequent]) / len(prediction_confidences[most_frequent])

        return most_frequent, avg_confidence
