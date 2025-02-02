"""
Tests for the server API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from server import app
import json

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "resources" in data
    assert "loaded_models" in data
    assert "total_models" in data

def test_chat_without_model():
    """Test chat endpoint without loading a model."""
    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model_id": "test-model",
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    assert response.status_code == 500
    assert "Invalid model ID: test-model" in response.json()["detail"]

@pytest.mark.asyncio
async def test_model_list():
    """Test getting the list of available models."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert len(data["models"]) > 0
    assert "default_model" in data
    assert "loaded_models" in data

@pytest.mark.asyncio
async def test_model_info():
    """Test getting information about a specific model."""
    # First get list of models
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    
    # Test with first model
    model_id = next(iter(data["models"]))
    model_info = data["models"][model_id]
    assert "name" in model_info
    assert "context_length" in model_info
    assert "temperature" in model_info
    assert "max_new_tokens" in model_info

    # Test with invalid model
    response = client.get("/models/invalid-model")
    assert response.status_code == 404
