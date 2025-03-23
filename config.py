import os

# Paths
DATASET_PATH = "dataset/asl_alphabet_train"
TEST_DATASET_PATH = "dataset/asl_alphabet_test"
MODEL_SAVE_DIR = "saved_models"
LOG_DIR = "logs"

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Model Parameters
NUM_LANDMARKS = 21  # MediaPipe hand landmarks
LANDMARK_DIM = 3    # x, y, z coordinates
INPUT_SHAPE = (NUM_LANDMARKS * LANDMARK_DIM,)
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
MAX_SAMPLES_PER_CLASS = 1000  # Limit samples to balance classes

# Preprocessing
IMAGE_SIZE = (200, 200)  # Original dataset image size
MIN_DETECTION_CONFIDENCE = 0.7

# Real-time Inference
PREDICTION_CONFIDENCE_THRESHOLD = 0.85
SMOOTHING_WINDOW_SIZE = 5  # Number of frames to average predictions
