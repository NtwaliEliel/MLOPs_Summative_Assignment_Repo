"""
Prediction utilities for trained model
"""

import numpy as np
from tensorflow import keras
from src.preprocessing import load_and_preprocess_image


class ModelPredictor:
    """
    Class for making predictions with trained model
    """
    
    def __init__(self, model_path='models/cnn_model.h5'):
        """
        Initialize predictor with model path
        
        Args:
            model_path: Path to saved model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model
        """
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_image(self, image_bytes):
        """
        Predict digit from image bytes
        
        Args:
            image_bytes: Image data in bytes
        
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Preprocess image
        processed_image = load_and_preprocess_image(image_bytes)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get all class probabilities
        probabilities = predictions[0].tolist()
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    def predict_batch(self, image_bytes_list):
        """
        Predict multiple images in batch
        
        Args:
            image_bytes_list: List of image bytes
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_bytes in image_bytes_list:
            result = self.predict_image(image_bytes)
            results.append(result)
        
        return results
    
    def reload_model(self):
        """
        Reload the model (useful after retraining)
        """
        self.load_model()


def predict_single_image(model, image_bytes):
    """
    Standalone function to predict single image
    
    Args:
        model: Loaded Keras model
        image_bytes: Image data in bytes
    
    Returns:
        Prediction dictionary
    """
    # Preprocess image
    processed_image = load_and_preprocess_image(image_bytes)
    
    # Make prediction
    predictions = model.predict(processed_image, verbose=0)
    
    # Get predicted class and confidence
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    probabilities = predictions[0].tolist()
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities
    }


def get_top_k_predictions(probabilities, k=3):
    """
    Get top K predictions with their probabilities
    
    Args:
        probabilities: Array of class probabilities
        k: Number of top predictions to return
    
    Returns:
        List of tuples (class, probability)
    """
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    top_k_predictions = [(int(idx), float(probabilities[idx])) for idx in top_k_indices]
    
    return top_k_predictions


def format_prediction_result(prediction_dict):
    """
    Format prediction result for display
    
    Args:
        prediction_dict: Dictionary with prediction results
    
    Returns:
        Formatted string
    """
    pred = prediction_dict['prediction']
    conf = prediction_dict['confidence']
    
    return f"Predicted Digit: {pred} (Confidence: {conf:.2%})"
