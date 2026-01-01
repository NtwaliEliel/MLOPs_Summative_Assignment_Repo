"""
CNN model architecture and training utilities
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create CNN model architecture for MNIST classification
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        
        # Third convolutional block
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=128, model_path='models/cnn_model.h5'):
    """
    Train the CNN model with callbacks
    
    Args:
        model: Keras model to train
        x_train: Training images
        y_train: Training labels
        x_val: Validation images
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_path: Path to save the best model
    
    Returns:
        Training history
    """
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model with multiple metrics
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def load_model(model_path):
    """
    Load a saved Keras model
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded Keras model
    """
    return keras.models.load_model(model_path)


def save_model(model, model_path):
    """
    Save Keras model to file
    
    Args:
        model: Keras model to save
        model_path: Path to save the model
    """
    model.save(model_path)
    print(f"Model saved to {model_path}")


def get_model_summary(model):
    """
    Get model architecture summary
    
    Args:
        model: Keras model
    
    Returns:
        Model summary as string
    """
    from io import StringIO
    import sys
    
    # Capture summary
    old_stdout = sys.stdout
    sys.stdout = summary_buffer = StringIO()
    model.summary()
    sys.stdout = old_stdout
    
    return summary_buffer.getvalue()


def retrain_model(model, new_x_train, new_y_train, epochs=10, batch_size=128, model_path='models/cnn_model.h5'):
    """
    Retrain existing model with new data
    
    Args:
        model: Existing Keras model
        new_x_train: New training images
        new_y_train: New training labels
        epochs: Number of retraining epochs
        batch_size: Batch size
        model_path: Path to save retrained model
    
    Returns:
        Training history
    """
    # Train on new data
    history = model.fit(
        new_x_train, new_y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )
    
    # Save retrained model
    model.save(model_path)
    
    return history
