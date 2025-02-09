"""
Tests for the model configuration module.
"""

import pytest
from config.models import (
    MODELS_CONFIG,
    DEFAULT_MODEL,
    get_model_config,
    get_total_required_space,
    validate_model_id,
    MAX_LOADED_MODELS
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
        "checksum_url",
        "trust_remote_code"
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
        assert isinstance(config["trust_remote_code"], bool)
        
        # Check size constraints
        assert config["size_gb"] <= 7, f"Model {model_id} exceeds 7GB size limit"
        
        # Check model ID format
        assert "/" in config["model_id"], f"Model {model_id} has invalid model_id format"
        
        # Check reasonable ranges
        assert 0.0 <= config["temperature"] <= 1.0, f"Model {model_id} has invalid temperature"
        assert 1024 <= config["context_length"] <= 32768, f"Model {model_id} has invalid context length"
        assert 100 <= config["max_new_tokens"] <= config["context_length"], \
            f"Model {model_id} has invalid max_new_tokens"

def test_default_model():
    """Test that DEFAULT_MODEL is valid."""
    assert DEFAULT_MODEL in MODELS_CONFIG
    config = get_model_config(DEFAULT_MODEL)
    assert config["name"] == MODELS_CONFIG[DEFAULT_MODEL]["name"]
    
    # Test default model size
    assert config["size_gb"] <= 7, "Default model exceeds 7GB size limit"

def test_get_model_config():
    """Test get_model_config function."""
    # Test valid model
    config = get_model_config(DEFAULT_MODEL)
    assert config == MODELS_CONFIG[DEFAULT_MODEL]
    
    # Test invalid model
    with pytest.raises(KeyError):
        get_model_config("nonexistent-model")
    
    # Test model with 4-bit quantization
    config = get_model_config("llama-3.2-3b")
    assert config["load_in_4bit"] is True, "Llama 3.2 3B should use 4-bit quantization"

def test_get_total_required_space():
    """Test get_total_required_space function."""
    total_space = get_total_required_space()
    assert total_space > 0
    assert total_space == sum(m["size_gb"] for m in MODELS_CONFIG.values())
    
    # Check that total space is reasonable given our constraints
    assert total_space <= len(MODELS_CONFIG) * 7, "Total space exceeds size constraints"

def test_validate_model_id():
    """Test validate_model_id function."""
    # Test valid models
    assert validate_model_id(DEFAULT_MODEL) is True
    assert validate_model_id("llama-3.2-3b") is True
    assert validate_model_id("deepseek-7b") is True
    assert validate_model_id("codellama-7b") is True
    assert validate_model_id("mistral-7b") is True
    
    # Test invalid models
    assert validate_model_id("nonexistent-model") is False
    assert validate_model_id("llama-13b") is False  # Too large
    assert validate_model_id("") is False  # Empty string
    assert validate_model_id(None) is False  # None value

def test_max_loaded_models():
    """Test MAX_LOADED_MODELS constant."""
    # Check that MAX_LOADED_MODELS is defined and reasonable
    assert MAX_LOADED_MODELS > 0, "MAX_LOADED_MODELS must be positive"
    assert MAX_LOADED_MODELS <= len(MODELS_CONFIG), "MAX_LOADED_MODELS cannot exceed total models"
    
    # Check memory constraints
    max_possible_memory = MAX_LOADED_MODELS * max(m["size_gb"] for m in MODELS_CONFIG.values())
    assert max_possible_memory <= 14, "MAX_LOADED_MODELS would allow too much memory usage"
