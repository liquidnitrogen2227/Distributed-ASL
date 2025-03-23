import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the preprocessed data."""
    try:
        X_train = np.load('data/processed/X_train.npy')
        X_val = np.load('data/processed/X_val.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_val = np.load('data/processed/y_val.npy')

        with open('data/processed/class_mapping.pkl', 'rb') as f:
            class_mapping = pickle.load(f)

        logger.info(f"Data loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation samples")
        logger.info(f"Number of classes: {len(class_mapping['class_to_idx'])}")

        return X_train, X_val, y_train, y_val, class_mapping
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def build_model(input_shape, num_classes):
    """
    Build a neural network model for hand landmark classification.

    Args:
        input_shape: Shape of the input features
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Input layer
        Dense(256, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(X_train, X_val, y_train, y_val, class_mapping):
    """
    Train the model on the provided data.

    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        class_mapping: Dictionary with class mappings
    """
    # Create timestamp for unique model directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(config.MODEL_SAVE_DIR, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    # Set up TensorBoard log directory
    log_dir = os.path.join(config.LOG_DIR, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(log_dir=log_dir)
    ]

    # Build model
    num_classes = len(class_mapping['class_to_idx'])
    model = build_model(input_shape=(X_train.shape[1],), num_classes=num_classes)

    # Display model summary
    model.summary()

    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save(os.path.join(model_dir, 'final_model.h5'))

    # Save model metadata
    with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
        pickle.dump({
            'class_mapping': class_mapping,
            'input_shape': (X_train.shape[1],),
            'training_history': history.history,
            'timestamp': timestamp
        }, f)

    # Plot training history
    plot_training_history(history, model_dir)

    logger.info(f"Model training completed. Model saved to {model_dir}")

    return model, model_dir

def plot_training_history(history, model_dir):
    """
    Plot and save training history.

    Args:
        history: Keras History object
        model_dir: Directory to save plots
    """
    # Accuracy plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))

def main():
    """Main function to load data and train the model."""
    # Load data
    X_train, X_val, y_train, y_val, class_mapping = load_data()

    # Train model
    model, model_dir = train_model(X_train, X_val, y_train, y_val, class_mapping)

    # Final evaluation
    test_loss, test_acc = model.evaluate(X_val, y_val)
    logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
