import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
import sys
import logging
from typing import Tuple, List, Dict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.landmark_utils import HandLandmarkExtractor, normalize_landmarks
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_image_folder(folder_path: str, class_name: str, extractor: HandLandmarkExtractor,
                        max_samples: int = None) -> Tuple[List[List[float]], List[str]]:
    """
    Process all images in a folder to extract hand landmarks.

    Args:
        folder_path: Path to the folder containing images
        class_name: Class label for these images
        extractor: HandLandmarkExtractor instance
        max_samples: Maximum number of samples to process per class

    Returns:
        Tuple of (landmarks_list, labels_list)
    """
    landmarks_list = []
    labels_list = []

    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Limit number of samples if specified
    if max_samples and len(image_files) > max_samples:
        image_files = image_files[:max_samples]

    for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
        image_path = os.path.join(folder_path, image_file)

        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not read image: {image_path}")
                continue

            # Extract landmarks
            result = extractor.extract_landmarks(image)

            if result:
                landmarks, _ = result

                # Normalize landmarks to make them invariant to scale and position
                normalized_landmarks = normalize_landmarks(landmarks)

                landmarks_list.append(normalized_landmarks)
                labels_list.append(class_name)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")

    return landmarks_list, labels_list

def create_dataset(dataset_path: str = config.DATASET_PATH,
                  max_samples_per_class: int = config.MAX_SAMPLES_PER_CLASS) -> Tuple[pd.DataFrame, Dict]:
    """
    Create a dataset of hand landmarks from the ASL alphabet images.

    Args:
        dataset_path: Path to the dataset folder
        max_samples_per_class: Maximum number of samples to use per class

    Returns:
        DataFrame with features and labels, and class mapping dictionary
    """
    logger.info(f"Creating dataset from {dataset_path}")

    all_landmarks = []
    all_labels = []

    # Count number of images per class for logging
    class_counts = {}

    with HandLandmarkExtractor(static_mode=True, min_detection_confidence=config.MIN_DETECTION_CONFIDENCE) as extractor:
        # Iterate through each class folder
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)

            if not os.path.isdir(class_path):
                continue

            # Process images in this class
            landmarks, labels = process_image_folder(
                class_path,
                class_name,
                extractor,
                max_samples_per_class
            )

            all_landmarks.extend(landmarks)
            all_labels.extend(labels)

            class_counts[class_name] = len(landmarks)
            logger.info(f"Processed {len(landmarks)} images for class {class_name}")

    # Convert to DataFrame
    columns = [f'landmark_{i}' for i in range(len(all_landmarks[0]))]
    df = pd.DataFrame(all_landmarks, columns=columns)
    df['label'] = all_labels

    # Create class mapping
    classes = sorted(df['label'].unique())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    logger.info(f"Dataset created with {len(df)} samples")
    logger.info(f"Class distribution: {class_counts}")

    return df, class_to_idx

def split_and_save_dataset(df: pd.DataFrame, class_to_idx: Dict,
                           test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Split dataset into train and validation sets and save them.

    Args:
        df: DataFrame with features and labels
        class_to_idx: Dictionary mapping class names to indices
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    """
    # Split features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].map(class_to_idx).values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save data
    os.makedirs('data/processed', exist_ok=True)

    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)

    # Save class mapping
    with open('data/processed/class_mapping.pkl', 'wb') as f:
        pickle.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': {i: cls for cls, i in class_to_idx.items()}
        }, f)

    logger.info(f"Data split and saved: {X_train.shape[0]} training, {X_val.shape[0]} validation samples")

def main():
    """Main function to create and save the dataset."""
    # Create dataset
    df, class_to_idx = create_dataset()

    # Split and save
    split_and_save_dataset(df, class_to_idx)

if __name__ == "__main__":
    main()
