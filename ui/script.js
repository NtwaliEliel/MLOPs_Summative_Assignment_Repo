// API base URL - change this to your deployed API URL
const API_URL = 'http://localhost:8000';

// Recent predictions storage
let recentPredictions = [];

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkHealth();
    setInterval(checkHealth, 30000); // Check health every 30 seconds
});

function initializeEventListeners() {
    // Predict upload area
    const predictUploadArea = document.getElementById('predict-upload-area');
    const predictFile = document.getElementById('predict-file');
    const predictBtn = document.getElementById('predict-btn');
    const removePredictBtn = document.getElementById('remove-predict');

    predictUploadArea.addEventListener('click', () => predictFile.click());
    predictFile.addEventListener('change', handlePredictFileSelect);
    predictBtn.addEventListener('click', handlePredict);
    removePredictBtn.addEventListener('click', removePredictImage);

    // Retrain upload area
    const retrainUploadArea = document.getElementById('retrain-upload-area');
    const retrainFiles = document.getElementById('retrain-files');
    const uploadBtn = document.getElementById('upload-btn');
    const retrainBtn = document.getElementById('retrain-btn');

    retrainUploadArea.addEventListener('click', () => retrainFiles.click());
    retrainFiles.addEventListener('change', handleRetrainFilesSelect);
    uploadBtn.addEventListener('click', handleUpload);
    retrainBtn.addEventListener('click', handleRetrain);

    // Drag and drop
    setupDragAndDrop(predictUploadArea, predictFile);
    setupDragAndDrop(retrainUploadArea, retrainFiles);
}

function setupDragAndDrop(area, fileInput) {
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.style.borderColor = 'var(--primary)';
    });

    area.addEventListener('dragleave', () => {
        area.style.borderColor = 'var(--border)';
    });

    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.style.borderColor = 'var(--border)';
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
    });
}

function handlePredictFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('predict-image').src = e.target.result;
            document.querySelector('#predict-upload-area .upload-placeholder').style.display = 'none';
            document.getElementById('predict-preview').style.display = 'block';
            document.getElementById('predict-btn').disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

function removePredictImage() {
    document.getElementById('predict-file').value = '';
    document.querySelector('#predict-upload-area .upload-placeholder').style.display = 'block';
    document.getElementById('predict-preview').style.display = 'none';
    document.getElementById('predict-btn').disabled = true;
    document.getElementById('prediction-result').style.display = 'none';
}

async function handlePredict() {
    const fileInput = document.getElementById('predict-file');
    const file = fileInput.files[0];

    if (!file) {
        showNotification('Please select an image', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const btn = document.getElementById('predict-btn');
    btn.disabled = true;
    btn.textContent = 'Predicting...';

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayPrediction(result);
        addToRecentPredictions(result);
    } catch (error) {
        showNotification('Prediction error: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Predict';
    }
}

function displayPrediction(result) {
    document.getElementById('predicted-digit').textContent = result.prediction;
    document.getElementById('confidence').textContent =
        `Confidence: ${(result.confidence * 100).toFixed(2)}%`;

    const probsContainer = document.getElementById('probabilities');
    probsContainer.innerHTML = '';

    result.probabilities.forEach((prob, index) => {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        probItem.innerHTML = `
            <span class="prob-label">${index}</span>
            <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
        `;
        probsContainer.appendChild(probItem);
    });

    document.getElementById('prediction-result').style.display = 'block';
}

function handleRetrainFilesSelect(e) {
    const files = e.target.files;
    const fileList = document.getElementById('file-list');
    fileList.innerHTML = '';

    if (files.length > 0) {
        Array.from(files).forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.textContent = file.name;
            fileList.appendChild(fileItem);
        });
        document.getElementById('upload-btn').disabled = false;
    }
}

async function handleUpload() {
    const filesInput = document.getElementById('retrain-files');
    const labelsInput = document.getElementById('labels-input');

    const files = filesInput.files;
    const labels = labelsInput.value.trim();

    if (files.length === 0) {
        showNotification('Please select images', 'error');
        return;
    }

    if (!labels) {
        showNotification('Please enter labels', 'error');
        return;
    }

    const formData = new FormData();
    Array.from(files).forEach(file => {
        formData.append('files', file);
    });
    formData.append('labels', labels);

    const btn = document.getElementById('upload-btn');
    btn.disabled = true;
    btn.textContent = 'Uploading...';

    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const result = await response.json();
        showNotification(result.message, 'success');

        // Clear inputs
        filesInput.value = '';
        labelsInput.value = '';
        document.getElementById('file-list').innerHTML = '';
    } catch (error) {
        showNotification('Upload error: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Upload Images';
    }
}

async function handleRetrain() {
    const btn = document.getElementById('retrain-btn');
    btn.disabled = true;
    btn.textContent = 'Retraining...';

    try {
        const response = await fetch(`${API_URL}/retrain`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error('Retraining failed');
        }

        const result = await response.json();
        displayRetrainResult(result);
        showNotification('Model retrained successfully!', 'success');
    } catch (error) {
        showNotification('Retrain error: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Retrain Model';
    }
}

function displayRetrainResult(result) {
    const retrainInfo = document.getElementById('retrain-info');
    retrainInfo.innerHTML = `
        <p><strong>Accuracy:</strong> ${(result.accuracy * 100).toFixed(2)}%</p>
        <p><strong>Epochs:</strong> ${result.epochs}</p>
        <p><strong>Training Samples:</strong> ${result.training_samples || 'N/A'}</p>
    `;
    document.getElementById('retrain-result').style.display = 'block';
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();

        document.getElementById('health-status').textContent = data.status;
        document.getElementById('health-status').style.color =
            data.status === 'healthy' ? 'var(--success)' : 'var(--danger)';

        const uptimeMinutes = Math.floor(data.uptime_seconds / 60);
        const uptimeHours = Math.floor(uptimeMinutes / 60);
        const uptimeDisplay = uptimeHours > 0
            ? `${uptimeHours}h ${uptimeMinutes % 60}m`
            : `${uptimeMinutes}m`;
        document.getElementById('uptime').textContent = uptimeDisplay;

        document.getElementById('model-status').textContent =
            data.model_loaded ? 'Loaded' : 'Not Loaded';
        document.getElementById('model-status').style.color =
            data.model_loaded ? 'var(--success)' : 'var(--warning)';
    } catch (error) {
        document.getElementById('health-status').textContent = 'Offline';
        document.getElementById('health-status').style.color = 'var(--danger)';
    }
}

function addToRecentPredictions(result) {
    const prediction = {
        digit: result.prediction,
        confidence: result.confidence,
        timestamp: new Date().toLocaleTimeString()
    };

    recentPredictions.unshift(prediction);
    if (recentPredictions.length > 10) {
        recentPredictions.pop();
    }

    updateRecentPredictionsList();
}

function updateRecentPredictionsList() {
    const container = document.getElementById('recent-predictions');

    if (recentPredictions.length === 0) {
        container.innerHTML = '<p class="empty-state">No predictions yet</p>';
        return;
    }

    container.innerHTML = recentPredictions.map(pred => `
        <div class="prediction-item">
            <span>Digit: <strong>${pred.digit}</strong></span>
            <span>${(pred.confidence * 100).toFixed(1)}%</span>
            <span class="text-secondary">${pred.timestamp}</span>
        </div>
    `).join('');
}

function showNotification(message, type = 'info') {
    // Simple alert for now - can be enhanced with toast notifications
    alert(message);
}
