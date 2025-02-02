#!/usr/bin/env python3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import HfApi
import hashlib
import requests
import json

# Load environment variables
print("Loading environment from .env file...")
env_path = os.path.join(os.path.dirname(__file__), '.env')
if not os.path.exists(env_path):
    raise ValueError(f"No .env file found at {env_path}")
print(f"Found .env file at: {env_path}")

load_dotenv(env_path)

# Get Hugging Face token
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not token:
    raise ValueError(
        "Please set HUGGING_FACE_HUB_TOKEN in your .env file.\n"
        "You can get your token from: https://huggingface.co/settings/tokens"
    )

# Model configurations - keep in sync with server.py
MODELS_CONFIG = {
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "context_length": 4096,
        "temperature": 0.7,
        "max_new_tokens": 500,
        "load_in_4bit": True,
        "checksum_url": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/raw/main/checksums.md5"
    },
    "deepseek-7b": {
        "name": "Deepseek Coder 7B",
        "model_id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "checksum_url": "https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5/raw/main/checksums.md5"
    },
    "codellama-7b": {
        "name": "CodeLlama 7B Instruct",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "checksum_url": "https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/raw/main/checksums.md5"
    },
    "codellama-13b": {
        "name": "CodeLlama 13B Instruct",
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "checksum_url": "https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/raw/main/checksums.md5"
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.3",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "checksum_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/raw/main/checksums.md5"
    },
    "deepseek-moe-16b": {
        "name": "Deepseek MoE 16B",
        "model_id": "deepseek-ai/deepseek-moe-16b-chat",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "trust_remote_code": True,
        "checksum_url": "https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/raw/main/checksums.md5"
    },
    "phi-2": {
        "name": "Microsoft Phi-2",
        "model_id": "microsoft/phi-2",
        "context_length": 2048,
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "load_in_4bit": True,
        "checksum_url": "https://huggingface.co/microsoft/phi-2/raw/main/checksums.md5"
    },
    "neural-chat-7b": {
        "name": "Neural Chat 7B",
        "model_id": "Intel/neural-chat-7b-v3-1",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "checksum_url": "https://huggingface.co/Intel/neural-chat-7b-v3-1/raw/main/checksums.md5"
    }
}

def verify_token():
    """Verify Hugging Face token and print user info"""
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print("✓ Token verified successfully")
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"\n❌ Token verification failed: {str(e)}")
        print("Please check your token in .env file")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return False

def check_model_files(model_id: str) -> bool:
    """Check if model exists in huggingface cache"""
    if model_id not in MODELS_CONFIG:
        return False
    
    config = MODELS_CONFIG[model_id]
    model_path = config["model_id"]
    
    try:
        # Try to load with local_files_only=True to check cache
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,  # Only check local cache
            trust_remote_code=config.get("trust_remote_code", False)
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,  # Only check local cache
            device_map="auto",
            trust_remote_code=config.get("trust_remote_code", False)
        )
        return True
    except Exception:
        return False

def verify_checksum(file_path: str, expected_md5: str) -> bool:
    """Verify the MD5 checksum of a file."""
    print(f"Verifying checksum for {os.path.basename(file_path)}...")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
        
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
            
    computed_md5 = md5_hash.hexdigest()
    is_valid = computed_md5 == expected_md5
    
    if is_valid:
        print(f"✅ Checksum verification passed for {os.path.basename(file_path)}")
    else:
        print(f"❌ Checksum verification failed for {os.path.basename(file_path)}")
        print(f"Expected: {expected_md5}")
        print(f"Got: {computed_md5}")
        
    return is_valid

def get_model_checksums(checksum_url: str) -> dict:
    """Get the MD5 checksums for model files from HuggingFace."""
    try:
        response = requests.get(checksum_url, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        
        checksums = {}
        for line in response.text.split('\n'):
            if line.strip():
                md5sum, filename = line.strip().split()
                checksums[filename] = md5sum
        return checksums
    except Exception as e:
        print(f"Warning: Could not fetch checksums from {checksum_url}: {str(e)}")
        return {}

def download_model(model_id: str):
    """Download a model to huggingface cache if not already present."""
    print(f"\nChecking/downloading model: {model_id}")
    
    try:
        # Get model config
        model_config = None
        for config in MODELS_CONFIG.values():
            if config["model_id"] == model_id:
                model_config = config
                break
                
        if not model_config:
            raise ValueError(f"Model {model_id} not found in MODELS_CONFIG")
            
        # Get checksums first
        checksums = get_model_checksums(model_config["checksum_url"])
        
        # Download and verify tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=True
        )
        
        # Verify tokenizer files
        tokenizer_files = [f for f in os.listdir(tokenizer.vocab_file) if f.endswith('.model')]
        for file in tokenizer_files:
            file_path = os.path.join(tokenizer.vocab_file, file)
            if file in checksums and not verify_checksum(file_path, checksums[file]):
                raise ValueError(f"Checksum verification failed for tokenizer file: {file}")
        
        # Download and verify model
        print("Downloading model...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_config.get("load_in_4bit", False),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Verify model files
        model_files = [f for f in os.listdir(model.config.name_or_path) if f.endswith('.bin')]
        for file in model_files:
            file_path = os.path.join(model.config.name_or_path, file)
            if file in checksums and not verify_checksum(file_path, checksums[file]):
                raise ValueError(f"Checksum verification failed for model file: {file}")
        
        print(f"✅ Successfully downloaded and verified {model_id}")
        return True
        
    except Exception as e:
        print(f"Error downloading model {model_id}: {str(e)}")
        return False

def main():
    """Check and download models"""
    print("\nVerifying token: " + token[:2] + "..." + token[-4:])
    if not verify_token():
        return
    
    print("\nChecking installed models...")
    
    # First check what's available
    available_models = []
    missing_models = []
    
    for model_id in MODELS_CONFIG:
        config = MODELS_CONFIG[model_id]
        if check_model_files(model_id):
            available_models.append(config["name"])
        else:
            missing_models.append((model_id, config["name"]))
    
    # Print status
    print("\nModel Status:")
    print("------------\n")
    
    if available_models:
        print("✓ Available models:")
        for name in available_models:
            print(f"  - {name}")
        print()
    
    if missing_models:
        print("❌ Missing models:")
        for _, name in missing_models:
            print(f"  - {name}")
        print()
        
        # Download missing models
        print("Downloading missing models...")
        for model_id, _ in missing_models:
            download_model(model_id)
    else:
        print("All models are already downloaded!")

if __name__ == "__main__":
    main()
