from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import torch
import logging
import traceback
import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import HfApi
import uvicorn
from dotenv import load_dotenv
import json

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("server")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disable Flash Attention and other optional features
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Model configurations
MODELS_CONFIG = {
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "context_length": 4096,
        "temperature": 0.7,
        "max_new_tokens": 500,
        "load_in_4bit": True,
        "size_gb": 3
    },
    "deepseek-7b": {
        "name": "Deepseek Coder 7B",
        "model_id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7
    },
    "codellama-7b": {
        "name": "CodeLlama 7B Instruct",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7
    },
    "codellama-13b": {
        "name": "CodeLlama 13B Instruct",
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 13
    },
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.3",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7
    },
    "deepseek-moe-16b": {
        "name": "Deepseek MoE 16B",
        "model_id": "deepseek-ai/deepseek-moe-16b-chat",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "trust_remote_code": True,
        "size_gb": 16
    },
    "phi-2": {
        "name": "Microsoft Phi-2",
        "model_id": "microsoft/phi-2",
        "context_length": 2048,
        "temperature": 0.7,
        "max_new_tokens": 1024,
        "load_in_4bit": True,
        "size_gb": 2.7
    },
    "neural-chat-7b": {
        "name": "Neural Chat 7B",
        "model_id": "Intel/neural-chat-7b-v3-1",
        "context_length": 8192,
        "temperature": 0.7,
        "max_new_tokens": 4096,
        "load_in_4bit": True,
        "size_gb": 7
    }
}

# Global cache for loaded models and tokenizers
MODELS = {}
TOKENIZERS = {}

def preload_models():
    """Preload only the default Llama 3.2 model"""
    logger.info("Starting model preloading...")
    
    try:
        # Only load Llama 3.2 initially
        preload_model("llama-3.2-3b")
        logger.info("\nSuccessfully loaded default model:")
        logger.info(f"  {MODELS_CONFIG['llama-3.2-3b']['name']}")
        return True
    except Exception as e:
        logger.error(f"Failed to load default model: {str(e)}")
        return False

def preload_model(model_id):
    """Preload a model"""
    if model_id not in MODELS_CONFIG:
        raise ValueError(f"Model {model_id} not found in config")
    
    config = MODELS_CONFIG[model_id]
    model_path = config["model_id"]
    
    logger.info(f"Preloading {model_id}...")
    logger.info("Loading tokenizer...")
    
    try:
        # Some models require trust_remote_code
        trust_remote_code = config.get("trust_remote_code", False)
        
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Try to load from cache
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=trust_remote_code
            )
            
            # Only use CPU offload for models >13GB since we have 24GB VRAM
            model_size = config.get("size_gb", 0)
            if model_size > 13:
                logger.info(f"Very large model detected ({model_size}GB), using CPU offload for {model_id}")
                max_memory = {0: "20GiB", "cpu": "48GiB"}
            else:
                # For smaller models, use most of GPU memory
                logger.info(f"Loading model {model_id} ({model_size}GB) entirely on GPU")
                max_memory = {0: "20GiB"}
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                device_map="auto",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
                max_memory=max_memory,
                offload_folder="offload"
            )
            
            MODELS[model_id] = {
                "model": model,
                "tokenizer": tokenizer,
                "config": config
            }
            
            logger.info(f"Successfully preloaded {model_id}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Failed to preload {model_id}: {str(e)}")
            raise
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Failed to preload {model_id}: {str(e)}")
        raise

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model_id: str = "phi-2"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

def format_messages(messages: List[ChatMessage], model_id: str):
    """Format messages for the model"""
    formatted_messages = []
    
    # Simple system message
    system_msg = {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
    formatted_messages.append(system_msg)
    
    # Add conversation history
    for msg in messages:
        formatted_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    return formatted_messages

def generate_response(messages: List[ChatMessage], model, tokenizer, model_id: str, max_new_tokens=None):
    """Generate response from model"""
    # Format messages for the model
    formatted_messages = format_messages(messages, model_id)
    
    # Build conversation string
    conversation = ""
    for msg in formatted_messages:
        if msg["role"] == "system":
            conversation += f"{msg['content']}\n\n"
        elif msg["role"] == "user":
            conversation += f"{msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"{msg['content']}\n"
    
    # Get model config
    config = MODELS_CONFIG[model_id]
    max_new_tokens = max_new_tokens or config.get("max_new_tokens", 500)
    temperature = config.get("temperature", 0.7)
    
    # Get context window from config or model
    context_window = config.get("context_length")
    if not context_window and hasattr(model.config, "max_position_embeddings"):
        context_window = model.config.max_position_embeddings
    if not context_window:
        # Default if not specified anywhere
        context_window = 2048
    
    # Reserve space for new tokens
    max_input_length = context_window - max_new_tokens
    
    # Generate response
    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get only the new content
    response = response[len(conversation):].strip()
    
    return response

@app.on_event("startup")
async def startup_event():
    """Server startup event - load models"""
    if not preload_models():
        logger.error("Failed to load models. Server cannot start.")
        raise RuntimeError("No models available")

@app.get("/api/models")
async def list_models():
    """List available models"""
    return {"models": [{"id": k, "name": v["name"]} for k, v in MODELS_CONFIG.items()]}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    loaded_models = [MODELS_CONFIG[model_id]["name"] for model_id in MODELS.keys()]
    return {
        "status": "ok",
        "loaded_models": loaded_models,
        "total_models": len(loaded_models)
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint"""
    try:
        model_id = request.model_id

        # Check if model exists in config
        if model_id not in MODELS_CONFIG:
            raise HTTPException(status_code=400, detail=f"Model {model_id} not found")
        
        # Load model if not already loaded
        if model_id not in MODELS:
            try:
                preload_model(model_id)
                logger.info(f"Loaded model {model_id} on demand")
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to load model {model_id}")

        model_data = MODELS[model_id]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Generate response
        response = generate_response(
            request.messages, 
            model, 
            tokenizer, 
            request.model_id,
            request.max_tokens
        )
        
        logger.info("Response generated successfully")
        
        # Add model name to response for context
        model_name = MODELS_CONFIG[request.model_id]["name"]
        return {"response": response, "model": model_name}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"Starting server with available models: {', '.join(MODELS_CONFIG.keys())}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)