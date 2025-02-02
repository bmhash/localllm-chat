"""
Tests for the model configuration module.
"""

import pytest
from config.models import (
    MODELS_CONFIG,
    DEFAULT_MODEL,
    get_model_config,
    get_total_required_space,
    validate_model_id
)

def test_models_config_structure():
    """Test that MODELS_CONFIG has the correct structure."""
    required_fields = {
        "name",
        "model_id",
        "context_length",
        "temperature",
        "max_new_tokens",
        "load_in_4bit",
        "size_gb",
        "checksum_url"
    }
    
    for model_id, config in MODELS_CONFIG.items():
        # Check all required fields are present
        assert all(field in config for field in required_fields), \
            f"Model {model_id} missing required fields"
            
        # Check field types
        assert isinstance(config["name"], str)
        assert isinstance(config["model_id"], str)
        assert isinstance(config["context_length"], int)
        assert isinstance(config["temperature"], float)
        assert isinstance(config["max_new_tokens"], int)
        assert isinstance(config["load_in_4bit"], bool)
        assert isinstance(config["size_gb"], (int, float))
        assert isinstance(config["checksum_url"], str)

def test_default_model():
    """Test that DEFAULT_MODEL is valid."""
    assert DEFAULT_MODEL in MODELS_CONFIG
    config = get_model_config(DEFAULT_MODEL)
    assert config["name"] == MODELS_CONFIG[DEFAULT_MODEL]["name"]

def test_get_model_config():
    """Test get_model_config function."""
    # Test valid model
    config = get_model_config(DEFAULT_MODEL)
    assert config == MODELS_CONFIG[DEFAULT_MODEL]
    
    # Test invalid model
    with pytest.raises(KeyError):
        get_model_config("nonexistent-model")

def test_get_total_required_space():
    """Test get_total_required_space function."""
    total_space = get_total_required_space()
    assert total_space > 0
    assert total_space == sum(m["size_gb"] for m in MODELS_CONFIG.values())

def test_validate_model_id():
    """Test validate_model_id function."""
    # Test valid model
    assert validate_model_id(DEFAULT_MODEL) is True
    
    # Test invalid model
    assert validate_model_id("nonexistent-model") is False
