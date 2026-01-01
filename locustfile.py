"""
Locust load testing script for MNIST API
"""

from locust import HttpUser, task, between
import io
from PIL import Image
import numpy as np


class MNISTUser(HttpUser):
    """
    Simulated user for load testing MNIST API
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """
        Called when a simulated user starts
        """
        # Generate a sample MNIST-like image
        self.test_image = self.generate_test_image()
    
    def generate_test_image(self):
        """
        Generate a random grayscale image for testing
        """
        # Create a 28x28 random grayscale image
        img_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    @task(10)
    def predict_image(self):
        """
        Test /predict endpoint (highest weight)
        """
        files = {'file': ('test.png', self.test_image, 'image/png')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    if 'prediction' in json_response and 'confidence' in json_response:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except Exception as e:
                    response.failure(f"Failed to parse JSON: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def check_health(self):
        """
        Test /health endpoint
        """
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    if 'status' in json_response and json_response['status'] == 'healthy':
                        response.success()
                    else:
                        response.failure("API not healthy")
                except Exception as e:
                    response.failure(f"Failed to parse JSON: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def get_stats(self):
        """
        Test /stats endpoint
        """
        with self.client.get("/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}")


class StressTestUser(HttpUser):
    """
    High-frequency user for stress testing
    """
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    def on_start(self):
        self.test_image = self.generate_test_image()
    
    def generate_test_image(self):
        img_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task
    def rapid_predictions(self):
        """
        Rapid-fire predictions for stress testing
        """
        files = {'file': ('test.png', self.test_image, 'image/png')}
        self.client.post("/predict", files=files)


# Custom load test scenarios
class LightLoadUser(HttpUser):
    """
    Light load scenario - 10 users
    """
    wait_time = between(2, 5)
    
    def on_start(self):
        self.test_image = self.generate_test_image()
    
    def generate_test_image(self):
        img_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task
    def predict(self):
        files = {'file': ('test.png', self.test_image, 'image/png')}
        self.client.post("/predict", files=files)


class MediumLoadUser(HttpUser):
    """
    Medium load scenario - 50 users
    """
    wait_time = between(1, 2)
    
    def on_start(self):
        self.test_image = self.generate_test_image()
    
    def generate_test_image(self):
        img_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task
    def predict(self):
        files = {'file': ('test.png', self.test_image, 'image/png')}
        self.client.post("/predict", files=files)


class HeavyLoadUser(HttpUser):
    """
    Heavy load scenario - 100+ users
    """
    wait_time = between(0.5, 1)
    
    def on_start(self):
        self.test_image = self.generate_test_image()
    
    def generate_test_image(self):
        img_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @task
    def predict(self):
        files = {'file': ('test.png', self.test_image, 'image/png')}
        self.client.post("/predict", files=files)
