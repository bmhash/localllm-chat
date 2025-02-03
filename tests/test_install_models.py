"""
Tests for model installation functionality.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from install_models import download_model, main
from config.models import MODELS_CONFIG

# Test data
TEST_MODEL_ID = "deepseek-7b"
TEST_TOKEN = "dummy_token"

@pytest.fixture
def mock_env():
    """Set up test environment variables."""
    with patch.dict(os.environ, {
        "HUGGING_FACE_HUB_TOKEN": TEST_TOKEN,
        "LOG_LEVEL": "INFO",
        "JSON_LOGGING": "true"
    }):
        yield

@pytest.fixture
def mock_cache_dir():
    """Create and clean up a temporary cache directory."""
    cache_dir = Path(".cache/huggingface")
    cache_dir.mkdir(parents=True, exist_ok=True)
    yield cache_dir
    # Clean up
    if cache_dir.exists():
        for file in cache_dir.glob("*"):
            file.unlink()
        cache_dir.rmdir()

def test_download_model_invalid_id(mock_env):
    """Test downloading with invalid model ID."""
    with patch("install_models.validate_model_id") as mock_validate:
        mock_validate.return_value = False
        assert download_model("invalid-model") is False

def test_download_model_insufficient_space(mock_env):
    """Test downloading with insufficient disk space."""
    with patch("install_models.validate_model_id") as mock_validate, \
         patch("install_models.check_disk_space") as mock_space:
        mock_validate.return_value = True
        mock_space.return_value = False
        assert download_model(TEST_MODEL_ID) is False

def test_download_model_no_access(mock_env, mock_cache_dir):
    """Test downloading when access is not granted."""
    with patch("install_models.validate_model_id") as mock_validate, \
         patch("install_models.check_disk_space") as mock_space, \
         patch("install_models.check_model_access") as mock_access, \
         patch("install_models.request_model_access") as mock_request:
        mock_validate.return_value = True
        mock_space.return_value = True
        mock_access.return_value = False
        mock_request.return_value = False
        
        assert download_model(TEST_MODEL_ID) is False

def test_download_model_access_timeout(mock_env, mock_cache_dir):
    """Test downloading when access request times out."""
    with patch("install_models.validate_model_id") as mock_validate, \
         patch("install_models.check_disk_space") as mock_space, \
         patch("install_models.check_model_access") as mock_access, \
         patch("install_models.request_model_access") as mock_request, \
         patch("install_models.wait_for_model_access") as mock_wait:
        mock_validate.return_value = True
        mock_space.return_value = True
        mock_access.return_value = False
        mock_request.return_value = True
        mock_wait.return_value = False
        
        assert download_model(TEST_MODEL_ID) is False

def test_download_model_success(mock_env, mock_cache_dir):
    """Test successful model download."""
    with patch("install_models.validate_model_id") as mock_validate, \
         patch("install_models.check_disk_space") as mock_space, \
         patch("install_models.check_model_access") as mock_access, \
         patch("install_models.get_model_files") as mock_files, \
         patch("install_models.download_file") as mock_download, \
         patch("install_models.verify_checksum") as mock_verify, \
         patch("install_models.login") as mock_login, \
         patch("pathlib.Path.stat") as mock_stat, \
         patch("pathlib.Path.is_dir") as mock_is_dir, \
         patch("pathlib.Path.exists") as mock_exists, \
         patch("pathlib.Path.mkdir") as mock_mkdir:
        mock_validate.return_value = True
        mock_space.return_value = True
        mock_access.return_value = True
        mock_verify.return_value = True
        mock_login.return_value = True
        mock_is_dir.return_value = True
        mock_exists.return_value = False  # File doesn't exist initially
        mock_mkdir.return_value = None

        # Mock file list
        mock_file = MagicMock()
        mock_file.rfilename = "model.bin"
        mock_file.size = 1000
        mock_files.return_value = {"files": [mock_file]}

        # Mock file stat for size verification after download
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1000
        mock_stat_result.st_mode = 0o100644  # Regular file mode
        mock_stat.return_value = mock_stat_result

        # Mock successful download
        mock_download.return_value = True

        assert download_model(TEST_MODEL_ID) is True
        mock_download.assert_called_once()

def test_download_model_checksum_mismatch(mock_env, mock_cache_dir):
    """Test download with checksum verification failure."""
    with patch("install_models.validate_model_id") as mock_validate, \
         patch("install_models.check_disk_space") as mock_space, \
         patch("install_models.check_model_access") as mock_access, \
         patch("install_models.get_model_files") as mock_files, \
         patch("install_models.download_file") as mock_download, \
         patch("install_models.verify_checksum") as mock_verify:
        mock_validate.return_value = True
        mock_space.return_value = True
        mock_access.return_value = True
        
        # Mock file list with checksum
        mock_file = MagicMock()
        mock_file.rfilename = "model.bin"
        mock_file.size = 1000
        mock_files.return_value = {"files": [mock_file]}
        
        # Mock successful download but failed verification
        mock_download.return_value = True
        mock_verify.return_value = False
        
        assert download_model(TEST_MODEL_ID, force=True) is False
        mock_download.assert_called_once()

def test_main_all_models_success(mock_env, mock_cache_dir):
    """Test main function with all models succeeding."""
    with patch("install_models.download_model") as mock_download, \
         patch("install_models.login") as mock_login, \
         patch("sys.argv", ["install_models.py"]):  # No args = download all models
        mock_download.return_value = True
        mock_login.return_value = True

        main()  # Should complete without error
        assert mock_download.call_count == len(MODELS_CONFIG)

def test_main_some_models_fail(mock_env, mock_cache_dir):
    """Test main function with some models failing."""
    with patch("install_models.download_model") as mock_download, \
         patch("install_models.login") as mock_login, \
         patch("sys.argv", ["install_models.py"]):  # No args = download all models
        # First model succeeds, others fail
        mock_download.side_effect = [True] + [False] * (len(MODELS_CONFIG) - 1)
        mock_login.return_value = True

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        
        # Should attempt all models despite failures
        assert mock_download.call_count == len(MODELS_CONFIG)

def test_main_single_model_success(mock_env, mock_cache_dir):
    """Test main function downloading a single model."""
    test_model = list(MODELS_CONFIG.keys())[0]
    with patch("install_models.download_model") as mock_download, \
         patch("install_models.login") as mock_login, \
         patch("sys.argv", ["install_models.py", "--model", test_model]):
        mock_download.return_value = True
        mock_login.return_value = True

        main()  # Should complete without error
        
        # Verify only the specified model was downloaded
        mock_download.assert_called_once_with(test_model, force=False)

def test_main_single_model_force(mock_env, mock_cache_dir):
    """Test main function force downloading a single model."""
    test_model = list(MODELS_CONFIG.keys())[0]
    with patch("install_models.download_model") as mock_download, \
         patch("install_models.login") as mock_login, \
         patch("sys.argv", ["install_models.py", "--model", test_model, "--force"]):
        mock_download.return_value = True
        mock_login.return_value = True

        main()  # Should complete without error
        
        # Verify model was force downloaded
        mock_download.assert_called_once_with(test_model, force=True)

def test_main_invalid_model(mock_env, mock_cache_dir):
    """Test main function with invalid model ID."""
    with patch("install_models.download_model") as mock_download, \
         patch("install_models.login") as mock_login, \
         patch("sys.argv", ["install_models.py", "--model", "invalid-model"]):
        mock_download.return_value = True
        mock_login.return_value = True

        # Should exit with error
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
