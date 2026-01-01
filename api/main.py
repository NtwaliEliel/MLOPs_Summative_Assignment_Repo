"""
FastAPI application for MNIST image classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
import time
import shutil
from typing import List
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import ModelPredictor
from src.model import load_model, retrain_model
from src.preprocessing import load_and_preprocess_image

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Image Classification API",
    description="API for handwritten digit classification and model retraining",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cnn_model.h5')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train')
predictor = None
start_time = time.time()

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """
    Initialize model on startup
    """
    global predictor
    try:
        predictor = ModelPredictor(MODEL_PATH)
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will need to be trained first")


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "MNIST Image Classification API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "upload": "/upload",
            "retrain": "/retrain",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint with uptime
    """
    uptime = time.time() - start_time
    
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "model_loaded": predictor is not None and predictor.model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Prediction result with confidence and probabilities
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Make prediction
        result = predictor.predict_image(image_bytes)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...), labels: str = Form(...)):
    """
    Upload multiple images for retraining
    
    Args:
        files: List of image files
        labels: Comma-separated labels corresponding to files
    
    Returns:
        Upload confirmation
    """
    try:
        # Parse labels
        label_list = [int(l.strip()) for l in labels.split(',')]
        
        if len(files) != len(label_list):
            raise HTTPException(status_code=400, detail="Number of files must match number of labels")
        
        # Save uploaded files
        saved_count = 0
        for file, label in zip(files, label_list):
            # Create label directory
            label_dir = os.path.join(DATA_DIR, str(label))
            os.makedirs(label_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(label_dir, f"{int(time.time())}_{file.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_count += 1
        
        return {
            "message": f"Uploaded {saved_count} images successfully",
            "count": saved_count
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post("/retrain")
async def retrain():
    """
    Retrain model with uploaded images
    
    Returns:
        Retraining result with metrics
    """
    global predictor
    
    try:
        # Check if there are uploaded images
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            raise HTTPException(status_code=400, detail="No training data available. Please upload images first.")
        
        # Load uploaded images
        images = []
        labels = []
        
        for label_dir in os.listdir(DATA_DIR):
            label_path = os.path.join(DATA_DIR, label_dir)
            if os.path.isdir(label_path):
                label = int(label_dir)
                
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    if os.path.isfile(img_path):
                        with open(img_path, 'rb') as f:
                            img_bytes = f.read()
                        
                        # Preprocess image
                        processed_img = load_and_preprocess_image(img_bytes)
                        images.append(processed_img[0])
                        labels.append(label)
        
        if len(images) == 0:
            raise HTTPException(status_code=400, detail="No valid images found for training")
        
        # Convert to numpy arrays
        x_train = np.array(images)
        y_train = np.array(labels)
        
        # Load existing model or create new one
        if predictor is None or predictor.model is None:
            from src.model import create_cnn_model
            model = create_cnn_model()
        else:
            model = predictor.model
        
        # Retrain model
        history = retrain_model(model, x_train, y_train, epochs=10, model_path=MODEL_PATH)
        
        # Reload model in predictor
        predictor = ModelPredictor(MODEL_PATH)
        
        # Get final accuracy
        final_accuracy = history.history['accuracy'][-1]
        
        return {
            "message": "Model retrained successfully",
            "accuracy": float(final_accuracy),
            "epochs": len(history.history['accuracy']),
            "training_samples": len(images)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    Get statistics about uploaded training data
    """
    try:
        stats = {}
        total_images = 0
        
        if os.path.exists(DATA_DIR):
            for label_dir in os.listdir(DATA_DIR):
                label_path = os.path.join(DATA_DIR, label_dir)
                if os.path.isdir(label_path):
                    count = len([f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))])
                    stats[label_dir] = count
                    total_images += count
        
        return {
            "total_images": total_images,
            "images_per_class": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
