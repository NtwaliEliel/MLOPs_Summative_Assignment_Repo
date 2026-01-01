"""
Image preprocessing utilities for MNIST dataset
"""

import numpy as np
from PIL import Image
import io


def load_and_preprocess_image(image_bytes, target_size=(28, 28)):
    """
    Load and preprocess an image from bytes
    
    Args:
        image_bytes: Image data in bytes
        target_size: Target size for resizing (height, width)
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to target size
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
    
    return img_array


def preprocess_mnist_data(x_train, x_test, y_train, y_test):
    """
    Preprocess MNIST dataset for training
    
    Args:
        x_train: Training images
        x_test: Test images
        y_train: Training labels
        y_test: Test labels
    
    Returns:
        Preprocessed training and test data
    """
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return x_train, x_test, y_train, y_test


def normalize_image(image_array):
    """
    Normalize image array to [0, 1] range
    
    Args:
        image_array: Input image array
    
    Returns:
        Normalized image array
    """
    return image_array.astype('float32') / 255.0


def resize_image(image, target_size=(28, 28)):
    """
    Resize image to target size
    
    Args:
        image: PIL Image object
        target_size: Target size (height, width)
    
    Returns:
        Resized PIL Image
    """
    return image.resize(target_size)


def convert_to_grayscale(image):
    """
    Convert image to grayscale
    
    Args:
        image: PIL Image object
    
    Returns:
        Grayscale PIL Image
    """
    return image.convert('L')


def augment_image(image_array):
    """
    Apply data augmentation to image
    
    Args:
        image_array: Input image array
    
    Returns:
        Augmented image array
    """
    # Random rotation
    from scipy.ndimage import rotate
    angle = np.random.uniform(-15, 15)
    augmented = rotate(image_array, angle, reshape=False)
    
    # Ensure values stay in valid range
    augmented = np.clip(augmented, 0, 1)
    
    return augmented


def batch_preprocess_images(image_files, target_size=(28, 28)):
    """
    Preprocess multiple images in batch
    
    Args:
        image_files: List of image file paths or bytes
        target_size: Target size for resizing
    
    Returns:
        Batch of preprocessed images
    """
    processed_images = []
    
    for img_file in image_files:
        if isinstance(img_file, bytes):
            img_array = load_and_preprocess_image(img_file, target_size)
        else:
            with open(img_file, 'rb') as f:
                img_bytes = f.read()
            img_array = load_and_preprocess_image(img_bytes, target_size)
        
        processed_images.append(img_array)
    
    # Stack all images into a single batch
    return np.vstack(processed_images)
