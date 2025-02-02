"""
Models Configuration Module

This module serves as the single source of truth for model configurations across the application.
Each model configuration contains all necessary parameters for model loading, resource management,
and operational settings.

Fields:
    name (str): Display name of the model
    model_id (str): HuggingFace model identifier
    context_length (int): Maximum context length the model supports
    temperature (float): Default sampling temperature
    max_new_tokens (int): Maximum number of tokens to generate
    load_in_4bit (bool): Whether to use 4-bit quantization
    size_gb (float): Approximate model size in gigabytes
    checksum_url (str): URL for model file MD5 checksums
    trust_remote_code (bool): Whether to trust remote code during model loading
"""

from typing import Dict, Any

MODELS_CONFIG: Dict[str, Dict[str, Any]] = {
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "context_length": 4096,
        "temperature": 0.7,
        "max_new_tokens": 500,
        "load_in_4bit": True,
        "size_gb": 3,
        "checksum_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/raw/main/checksums.md5",
        "trust_remote_code": False
    },
    "deepseek-7b": {
        "name": "Deepseek Coder 7B",
        "model_id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7,
        "checksum_url": "https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5/raw/main/checksums.md5",
        "trust_remote_code": False
    },
    "codellama-7b": {
        "name": "CodeLlama 7B Instruct",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7,
        "checksum_url": "https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/raw/main/checksums.md5",
        "trust_remote_code": False
    },
    "codellama-13b": {
        "name": "CodeLlama 13B Instruct",
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 13,
        "checksum_url": "https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/raw/main/checksums.md5",
        "trust_remote_code": False
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.3",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7,
        "checksum_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/checksums.md5",
        "trust_remote_code": False
    },
    "deepseek-moe-16b": {
        "name": "Deepseek MoE 16B",
        "model_id": "deepseek-ai/deepseek-moe-16b-chat",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 16,
        "checksum_url": "https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/raw/main/checksums.md5",
        "trust_remote_code": True
    },
    "phi-2": {
        "name": "Microsoft Phi-2",
        "model_id": "microsoft/phi-2",
        "context_length": 2048,
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "load_in_4bit": True,
        "size_gb": 2.7,
        "checksum_url": "https://huggingface.co/microsoft/phi-2/raw/main/checksums.md5",
        "trust_remote_code": False
    },
    "neural-chat-7b": {
        "name": "Neural Chat 7B",
        "model_id": "Intel/neural-chat-7b-v3-1",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7,
        "checksum_url": "https://huggingface.co/Intel/neural-chat-7b-v3-1/raw/main/checksums.md5",
        "trust_remote_code": False
    }
}

# Default model to load at startup
DEFAULT_MODEL = "llama-3.2-3b"

def get_model_config(model_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_id: The identifier of the model to get configuration for
        
    Returns:
        Dict containing the model's configuration
        
    Raises:
        KeyError: If model_id is not found in configurations
    """
    if model_id not in MODELS_CONFIG:
        raise KeyError(f"Model {model_id} not found in configurations")
    return MODELS_CONFIG[model_id]

def get_total_required_space() -> float:
    """
    Calculate total disk space required for all models.
    
    Returns:
        Float representing total space needed in gigabytes
    """
    return sum(config["size_gb"] for config in MODELS_CONFIG.values())

def validate_model_id(model_id: str) -> bool:
    """
    Validate if a model ID exists in configurations.
    
    Args:
        model_id: The identifier to validate
        
    Returns:
        Boolean indicating if model_id is valid
    """
    return model_id in MODELS_CONFIG
