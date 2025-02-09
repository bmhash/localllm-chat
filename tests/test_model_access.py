"""
Tests for model access functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from huggingface_hub.utils import HfHubHTTPError
import requests
import torch

from utils.model_access import check_model_access, request_model_access, wait_for_model_access
from config.models import MODELS_CONFIG

# Test data
TEST_MODEL_ID = "llama-3.2-3b"  # Updated to use a model we actually have
TEST_TOKEN = "dummy_token"

@pytest.fixture
def mock_env():
    """Set up test environment variables."""
    with patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": TEST_TOKEN}):
        yield

@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = MagicMock()
    response.status_code = 200
    return response

def test_check_model_access_has_access(mock_env):
    """Test checking model access when user has access."""
    with patch("utils.model_access.model_info") as mock_info:
        mock_info.return_value = {"id": TEST_MODEL_ID}
        assert check_model_access(TEST_MODEL_ID) is True
        mock_info.assert_called_once_with(TEST_MODEL_ID)

def test_check_model_access_unauthorized(mock_env):
    """Test checking model access when user is unauthorized."""
    with patch("utils.model_access.model_info") as mock_info:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_info.side_effect = HfHubHTTPError("Unauthorized", mock_response)
        assert check_model_access(TEST_MODEL_ID) is False
        mock_info.assert_called_once_with(TEST_MODEL_ID)

def test_check_model_access_rejected(mock_env):
    """Test checking model access when user is rejected."""
    with patch("utils.model_access.model_info") as mock_info:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_info.side_effect = HfHubHTTPError("Forbidden", mock_response)
        assert check_model_access(TEST_MODEL_ID) is False
        mock_info.assert_called_once_with(TEST_MODEL_ID)

def test_check_model_access_error(mock_env):
    """Test checking model access with unexpected error."""
    with patch("utils.model_access.model_info") as mock_info:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_info.side_effect = HfHubHTTPError("Server Error", mock_response)
        with pytest.raises(HfHubHTTPError):
            check_model_access(TEST_MODEL_ID)

def test_request_model_access_already_has_access(mock_env):
    """Test requesting access when user already has access."""
    with patch("utils.model_access.check_model_access") as mock_check:
        mock_check.return_value = True
        assert request_model_access(TEST_MODEL_ID) is True
        mock_check.assert_called_once_with(TEST_MODEL_ID)

def test_request_model_access_auto_approved(mock_env):
    """Test requesting access with automatic approval."""
    with patch("utils.model_access.check_model_access") as mock_check, \
         patch("utils.model_access.HfApi") as mock_api:
        # First check fails, second check passes (after request)
        mock_check.side_effect = [False, True]
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance
        
        assert request_model_access(TEST_MODEL_ID) is True
        assert mock_check.call_count == 2
        mock_api_instance.request_model_access.assert_called_once_with(TEST_MODEL_ID)

def test_request_model_access_manual_approval(mock_env):
    """Test requesting access with manual approval."""
    with patch("utils.model_access.check_model_access") as mock_check, \
         patch("utils.model_access.HfApi") as mock_api:
        # Access remains False after request
        mock_check.return_value = False
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance
        
        assert request_model_access(TEST_MODEL_ID) is False
        assert mock_check.call_count == 2
        mock_api_instance.request_model_access.assert_called_once_with(TEST_MODEL_ID)

def test_request_model_access_error(mock_env):
    """Test requesting access with API error."""
    with patch("utils.model_access.check_model_access") as mock_check, \
         patch("utils.model_access.HfApi") as mock_api:
        mock_check.return_value = False
        mock_api_instance = MagicMock()
        mock_api_instance.request_model_access.side_effect = Exception("API Error")
        mock_api.return_value = mock_api_instance
        
        assert request_model_access(TEST_MODEL_ID) is False
        mock_check.assert_called_once_with(TEST_MODEL_ID)
        mock_api_instance.request_model_access.assert_called_once_with(TEST_MODEL_ID)

def test_wait_for_model_access_immediate(mock_env):
    """Test waiting for access when immediately granted."""
    with patch("utils.model_access.check_model_access") as mock_check, \
         patch("utils.model_access.time.sleep") as mock_sleep:
        mock_check.return_value = True
        assert wait_for_model_access(TEST_MODEL_ID) is True
        mock_check.assert_called_once_with(TEST_MODEL_ID)
        mock_sleep.assert_not_called()

def test_wait_for_model_access_delayed(mock_env):
    """Test waiting for access when granted after delay."""
    with patch("utils.model_access.check_model_access") as mock_check, \
         patch("utils.model_access.time.sleep") as mock_sleep, \
         patch("utils.model_access.time.time") as mock_time:
        # Access granted after 2 checks
        mock_check.side_effect = [False, False, True]
        # Simulate time passing - provide enough values for all calls including logging
        mock_time.side_effect = [0, 0, 10, 10, 20, 20, 30, 30]

        assert wait_for_model_access(TEST_MODEL_ID, timeout=60, check_interval=10) is True
        assert mock_check.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(10)

def test_wait_for_model_access_timeout(mock_env):
    """Test waiting for access when timeout occurs."""
    with patch("utils.model_access.check_model_access") as mock_check, \
         patch("utils.model_access.time.sleep") as mock_sleep, \
         patch("utils.model_access.time.time") as mock_time:
        mock_check.return_value = False
        # Simulate time passing beyond timeout - provide enough values for all calls including logging
        mock_time.side_effect = [0, 0, 100, 100, 200, 200]

        assert wait_for_model_access(TEST_MODEL_ID, timeout=60, check_interval=10) is False
        assert mock_check.call_count == 1
