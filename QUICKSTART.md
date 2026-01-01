# Quick Start Guide

## Prerequisites
- Python 3.10+
- pip

## Installation

1. **Install dependencies**
```bash
cd api
pip install -r requirements.txt
```

2. **Train the model**
Open and run `notebook/image_mlops.ipynb` to train the model and generate `models/cnn_model.h5`

3. **Start the API**
```bash
cd api
python main.py
```

4. **Open the UI**
Open `ui/index.html` in your browser or serve it:
```bash
cd ui
python -m http.server 3000
```

## Testing

### Load Testing
```bash
pip install locust
locust -f locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 and configure:
- Users: 10, 50, 100, or 200
- Spawn rate: 10

## Deployment

### Render
1. Push to GitHub
2. Create new Web Service on Render
3. Connect repository
4. Deploy automatically using `render.yaml`

### Docker
```bash
cd api
docker build -t mnist-mlops .
docker run -p 8000:8000 mnist-mlops
```

## API Endpoints

- `POST /predict` - Predict digit from image
- `POST /upload` - Upload training images
- `POST /retrain` - Retrain model
- `GET /health` - Health check
- `GET /stats` - Training data statistics

## Project Structure

```
image-mlops-pipeline/
├── README.md
├── notebook/
│   └── image_mlops.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── ui/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── cnn_model.h5
├── locustfile.py
└── render.yaml
```
