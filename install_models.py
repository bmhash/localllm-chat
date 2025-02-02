#!/usr/bin/env python3
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv
from huggingface_hub import HfApi

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
        "load_in_4bit": True
    },
    "deepseek-7b": {
        "name": "Deepseek Coder 7B",
        "model_id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True
    },
    "codellama-7b": {
        "name": "CodeLlama 7B Instruct",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True
    },
    "codellama-13b": {
        "name": "CodeLlama 13B Instruct",
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.3",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True
    },
    "deepseek-moe-16b": {
        "name": "Deepseek MoE 16B",
        "model_id": "deepseek-ai/deepseek-moe-16b-chat",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "trust_remote_code": True
    },
    "phi-2": {
        "name": "Microsoft Phi-2",
        "model_id": "microsoft/phi-2",
        "context_length": 2048,
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "load_in_4bit": True
    },
    "neural-chat-7b": {
        "name": "Neural Chat 7B",
        "model_id": "Intel/neural-chat-7b-v3-1",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True
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

def download_model(model_id: str):
    """Download a model to huggingface cache if not already present"""
    if model_id not in MODELS_CONFIG:
        print(f"Model {model_id} not found in config")
        return False
    
    config = MODELS_CONFIG[model_id]
    model_path = config["model_id"]
    
    # Skip if already in cache
    if check_model_files(model_id):
        print(f"✓ {config['name']} already in cache")
        return True
    
    print(f"\nDownloading {config['name']} ({model_path})...")
    
    try:
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        
        # Some models require trust_remote_code
        trust_remote_code = config.get("trust_remote_code", False)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=token,
                trust_remote_code=trust_remote_code
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=token,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code
            )
            print(f"✓ Successfully downloaded {config['name']}")
            return True
            
        except Exception as e:
            if "gated repo" in str(e).lower():
                print(f"\n⚠️  {config['name']} requires access approval:")
                print(f"Please visit https://huggingface.co/{model_path}")
                print("Click 'Access Request' to request access.")
                print("Once approved, your token will automatically work.\n")
            else:
                print(f"❌ Error downloading {config['name']}: {str(e)}")
            return False
        
    except Exception as e:
        print(f"❌ Error downloading {config['name']}: {str(e)}")
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
