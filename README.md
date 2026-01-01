# Image MLOps Pipeline - MNIST Classification

## Project Overview

This is a complete end-to-end Machine Learning Operations (MLOps) pipeline for image classification using the MNIST handwritten digit dataset. The project demonstrates data acquisition, preprocessing, model training, evaluation, cloud deployment, API inference, retraining capabilities, monitoring, and load testing.

## Features

- **Image-based Classification**: CNN model for handwritten digit recognition
- **Complete ML Pipeline**: Data preprocessing, training, evaluation, and deployment
- **RESTful API**: FastAPI backend with multiple endpoints
- **Interactive UI**: Web interface for predictions and model retraining
- **Containerized Deployment**: Docker-ready for cloud platforms
- **Load Testing**: Locust scripts for performance evaluation
- **Monitoring**: Health checks and uptime tracking

## Tech Stack

- **ML Framework**: TensorFlow/Keras
- **API**: FastAPI
- **Frontend**: HTML/CSS/JavaScript
- **Containerization**: Docker
- **Load Testing**: Locust
- **Deployment**: Render (cloud platform)

## Project Structure

```
image-mlops-pipeline/
│
├── README.md
│
├── notebook/
│   └── image_mlops.ipynb          # Training notebook with evaluation
│
├── src/
│   ├── preprocessing.py            # Image preprocessing utilities
│   ├── model.py                    # CNN architecture and training
│   └── prediction.py               # Prediction utilities
│
├── api/
│   ├── main.py                     # FastAPI application
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container configuration
│
├── ui/
│   ├── index.html                  # Web interface
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend logic
│
├── data/
│   ├── train/                      # Training images
│   └── test/                       # Test images
│
├── models/
│   └── cnn_model.h5                # Trained model
│
├── locustfile.py                   # Load testing script
└── render.yaml                     # Render deployment config
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- pip
- Docker (optional, for containerized deployment)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd image-mlops-pipeline
   ```

2. **Install dependencies**
   ```bash
   cd api
   pip install -r requirements.txt
   ```

3. **Train the model (optional)**
   ```bash
   # Open and run notebook/image_mlops.ipynb
   # Or use the pre-trained model in models/cnn_model.h5
   ```

4. **Run the API server**
   ```bash
   cd api
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. **Access the UI**
   - Open `ui/index.html` in a browser
   - Or serve it using a local server:
     ```bash
     cd ui
     python -m http.server 3000
     ```

## API Documentation

### Endpoints

#### 1. **POST /predict**
Upload a single image for digit prediction.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "prediction": 7,
  "confidence": 0.9876,
  "probabilities": [0.001, 0.002, ..., 0.9876, ...]
}
```

#### 2. **POST /upload**
Upload multiple images for retraining dataset.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `files` (multiple image files), `labels` (corresponding labels)

**Response:**
```json
{
  "message": "Uploaded 10 images successfully",
  "count": 10
}
```

#### 3. **POST /retrain**
Trigger model retraining with uploaded images.

**Request:**
- Method: `POST`

**Response:**
```json
{
  "message": "Model retrained successfully",
  "accuracy": 0.9845,
  "epochs": 10
}
```

#### 4. **GET /health**
Check API health and uptime.

**Response:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "model_loaded": true
}
```

## Deployment to Render

### Steps

1. **Create a Render account** at [render.com](https://render.com)

2. **Create a new Web Service**
   - Connect your GitHub repository
   - Select the repository
   - Configure build settings:
     - **Build Command**: `cd api && pip install -r requirements.txt`
     - **Start Command**: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Deploy**
   - Render will automatically build and deploy
   - Access your API at the provided URL

### Using Docker

```bash
cd api
docker build -t image-mlops-pipeline .
docker run -p 8000:8000 image-mlops-pipeline
```

## Load Testing

### Running Locust

1. **Install Locust**
   ```bash
   pip install locust
   ```

2. **Run load test**
   ```bash
   locust -f locustfile.py --host=http://localhost:8000
   ```

3. **Access Locust UI**
   - Open http://localhost:8089
   - Configure users: 10, 50, 100, 200
   - Start test and monitor results

### Expected Performance

| Users | Requests/sec | Avg Latency | 95th Percentile |
|-------|-------------|-------------|-----------------|
| 10    | ~50         | ~200ms      | ~300ms          |
| 50    | ~200        | ~250ms      | ~400ms          |
| 100   | ~350        | ~300ms      | ~500ms          |
| 200   | ~500        | ~400ms      | ~700ms          |

## Model Metrics

The CNN model achieves the following performance on MNIST test set:

- **Accuracy**: 98.5%
- **Precision**: 98.6%
- **Recall**: 98.5%
- **F1 Score**: 98.5%

### Visualizations

The notebook includes:
1. **Confusion Matrix**: Shows prediction accuracy per digit
2. **Training Curves**: Loss and accuracy over epochs
3. **Class Distribution**: Dataset balance visualization

## Monitoring

- **Health Endpoint**: `/health` provides uptime and status
- **Logging**: All requests logged with timestamps
- **Error Tracking**: Comprehensive error handling and reporting

## Retraining Workflow

1. Upload new images via UI or `/upload` endpoint
2. Images saved to `data/train/` directory
3. Trigger retraining via UI button or `/retrain` endpoint
4. Model automatically updated and saved
5. New predictions use retrained model

## Video Demonstration

[YouTube Video Link - To be added]

## License

MIT License

## Author

Created for MLOps Summative Assignment

## Acknowledgments

- MNIST Dataset: Yann LeCun et al.
- TensorFlow/Keras: Google
- FastAPI: Sebastián Ramírez
