"""
Tests for the server API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks
from unittest.mock import patch, MagicMock
from server import app, load_model
import json
import torch

# Create test client
client = TestClient(app)

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "Test response"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer

@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_chat_without_model():
    """Test chat endpoint without loading a model."""
    response = client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model_id": "test-model"
        }
    )
    assert response.status_code == 500
    assert "detail" in response.json()

def test_model_list():
    """Test getting the list of available models."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "loaded_models" in data
    assert "default_model" in data

def test_model_loading(mock_tokenizer, mock_model):
    """Test model loading with mocked transformers."""
    with patch('server.AutoTokenizer.from_pretrained') as mock_tokenizer_cls, \
         patch('server.AutoModelForCausalLM.from_pretrained') as mock_model_cls:
        
        mock_tokenizer_cls.return_value = mock_tokenizer
        mock_model_cls.return_value = mock_model
        
        # Test loading a model
        background_tasks = BackgroundTasks()
        model, tokenizer = load_model("llama-3.2-3b", background_tasks)
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer

def test_chat_with_context(mock_tokenizer, mock_model):
    """Test chat endpoint with conversation context."""
    with patch('server.AutoTokenizer.from_pretrained') as mock_tokenizer_cls, \
         patch('server.AutoModelForCausalLM.from_pretrained') as mock_model_cls:

        # Configure mock tokenizer
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        
        # Mock tensor with proper operations
        class MockTensor:
            def __init__(self, tensor):
                self.tensor = tensor
                self.shape = tensor.shape
                
            def to(self, device):
                return self
                
            def __getitem__(self, idx):
                if isinstance(idx, tuple):
                    return self.tensor[idx[0]][idx[1]]
                if isinstance(idx, slice):
                    return self.tensor[idx]
                return self.tensor[idx]
                
            def __len__(self):
                return len(self.tensor)
                
            def size(self, dim=None):
                if dim is not None:
                    return self.shape[dim]
                return self.shape

        # Create mock tensors for input and output
        input_tensor = MockTensor(input_ids)
        attention_tensor = MockTensor(attention_mask)
        output_tensor = MockTensor(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]))
        
        # Create a callable mock tokenizer that preserves inputs
        class MockTokenizer:
            def __init__(self):
                self.inputs = None
                
            def __call__(self, text, return_tensors=None, padding=None):
                self.inputs = {
                    "input_ids": input_tensor,
                    "attention_mask": attention_tensor
                }
                return self.inputs
                
            def decode(self, tokens, skip_special_tokens=True):
                if isinstance(tokens, MockTensor):
                    tokens = tokens.tensor
                return "I'm doing well, thanks for asking!"
                
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2
            
        mock_tokenizer = MockTokenizer()
        mock_tokenizer_cls.return_value = mock_tokenizer

        # Configure mock model
        mock_model.device = torch.device("cpu")
        mock_model.generate = MagicMock(return_value=output_tensor)
        mock_model_cls.return_value = mock_model

        # Load model
        background_tasks = BackgroundTasks()
        model, tokenizer = load_model("llama-3.2-3b", background_tasks)

        # Test chat with context
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"}
                ],
                "model_id": "llama-3.2-3b"
            }
        )

        assert response.status_code == 200
        assert "response" in response.json()
