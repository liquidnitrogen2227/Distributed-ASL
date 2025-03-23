import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_evaluation_data():
    """Load validation data for model evaluation."""
    try:
        X_val = np.load('data/processed/X_val.npy')
        y_val = np.load('data/processed/y_val.npy')

        with open('data/processed/class_mapping.pkl', 'rb') as f:
            class_mapping = pickle.load(f)

        return X_val, y_val, class_mapping
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}")
        raise

def load_latest_model():
    """Load the latest trained model based on directory timestamp."""
    model_dirs = [d for d in os.listdir(config.MODEL_SAVE_DIR) if os.path.isdir(os.path.join(config.MODEL_SAVE_DIR, d))]

    if not model_dirs:
        logger.error("No trained models found.")
        return None, None

    # Sort by timestamp (assuming directory names are model_YYYYMMDD-HHMMSS)
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(config.MODEL_SAVE_DIR, latest_model_dir, 'best_model.h5')

    if not os.path.exists(model_path):
        model_path = os.path.join(config.MODEL_SAVE_DIR, latest_model_dir, 'final_model.h5')

    if not os.path.exists(model_path):
        logger.error(f"No model file found in {latest_model_dir}")
        return None, None

    # Load model info
    info_path = os.path.join(config.MODEL_SAVE_DIR, latest_model_dir, 'model_info.pkl')
    if os.path.exists(info_path):
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
    else:
        model_info = None

    # Load model
    model = load_model(model_path)
    logger.info(f"Loaded model from {model_path}")

    return model, model_info

def evaluate_model(model, X_val, y_val, class_mapping):
    """
    Evaluate the model performance on validation data.

    Args:
        model: Loaded Keras model
        X_val: Validation features
        y_val: Validation labels
        class_mapping: Dictionary with class mappings
    """
    # Get predictions
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_val)
    logger.info(f"Validation accuracy: {accuracy:.4f}")

    # Generate classification report
    idx_to_class = class_mapping['idx_to_class']
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    report = classification_report(y_val, y_pred, target_names=class_names)
    logger.info(f"Classification Report:\n{report}")

    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    return y_pred, cm, class_names

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(20, 16))

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()

    # Save plot
    os.makedirs('evaluation_results', exist_ok=True)
    plt.savefig('evaluation_results/confusion_matrix.png')
    plt.close()

def analyze_mistakes(y_val, y_pred, X_val, class_mapping, top_n=5):
    """
    Analyze the most common misclassifications.

    Args:
        y_val: True labels
        y_pred: Predicted labels
        X_val: Validation features
        class_mapping: Dictionary with class mappings
        top_n: Number of most common mistakes to report
    """
    idx_to_class = class_mapping['idx_to_class']

    # Find misclassified samples
    misclassified = y_val != y_pred

    if not np.any(misclassified):
        logger.info("No misclassifications found!")
        return

    # Get indices of misclassified samples
    misclassified_indices = np.where(misclassified)[0]

    # Create pairs of (true, predicted) labels
    true_pred_pairs = [(y_val[i], y_pred[i]) for i in misclassified_indices]

    # Count occurrences of each misclassification type
    pair_counts = {}
    for true_label, pred_label in true_pred_pairs:
        pair = (true_label, pred_label)
        if pair not in pair_counts:
            pair_counts[pair] = 0
        pair_counts[pair] += 1

    # Sort by frequency
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    # Report top N most common mistakes
    logger.info(f"Top {top_n} most common misclassifications:")
    for (true_label, pred_label), count in sorted_pairs[:top_n]:
        true_class = idx_to_class[true_label]
        pred_class = idx_to_class[pred_label]
        logger.info(f"  {true_class} misclassified as {pred_class}: {count} instances")

def main():
    """Main function to evaluate the model."""
    # Load validation data
    X_val, y_val, class_mapping = load_evaluation_data()

    # Load model
    model, model_info = load_latest_model()
    if model is None:
        return

    # Evaluate model
    y_pred, cm, class_names = evaluate_model(model, X_val, y_val, class_mapping)

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)

    # Analyze common mistakes
    analyze_mistakes(y_val, y_pred, X_val, class_mapping)

    logger.info("Evaluation completed. Results saved to 'evaluation_results' directory.")

if __name__ == "__main__":
    main()
