#!/bin/bash

# Configuration
PYTHON_EXEC="/usr/local/bin/python3.11"
VENV_DIR=".venv_local"
API_DIR="api"
UI_DIR="ui"

echo "🚀 Starting MNIST MLOps Project local setup..."

# Check if Python 3.11 exists
if ! [ -x "$PYTHON_EXEC" ]; then
    echo "❌ Error: Python 3.11 not found at $PYTHON_EXEC"
    echo "Please ensure Python 3.11 is installed. You can install it via Homebrew: brew install python@3.11"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment with Python 3.11..."
    $PYTHON_EXEC -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r $API_DIR/requirements.txt

# Start the Backend
echo "🌐 Starting FastAPI Backend on http://localhost:8000..."
# Added --reload to pick up code changes automatically
cd $API_DIR
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
sleep 5

# Check health
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is healthy!"
else
    echo "⚠️ Backend might still be starting or failed. Please check the logs."
fi

echo ""
echo "🎉 Setup complete!"
echo "--------------------------------------------------"
echo "💻 Access the API Documentation: http://localhost:8000/docs"
echo "🖥️ Access the Web UI:"
echo "   Option 1: Open the file directly: open $UI_DIR/index.html"
echo "   Option 2: Run a local server for the UI:"
echo "             cd $UI_DIR && python3 -m http.server 3000"
echo "--------------------------------------------------"
echo "Press Ctrl+C to stop the backend."

# Keep the script running to catch Ctrl+C
trap "kill $BACKEND_PID; exit" INT
wait $BACKEND_PID
