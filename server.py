"""
LocalLLM Chat Server

This module provides a FastAPI server for the LocalLLM Chat interface.
It handles model management, chat interactions, and provides health monitoring endpoints.

Features:
- Dynamic model loading and unloading
- Context preservation during model switching
- Resource usage monitoring
- Performance metrics tracking
- Structured logging
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Tuple
import torch
import os
import psutil
import shutil
from pathlib import Path
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from config.models import (
    MODELS_CONFIG,
    DEFAULT_MODEL,
    get_model_config,
    validate_model_id
)
from utils.logging import setup_logging, metrics
import logging
from chat_formatting import format_chat_messages

# Constants
MAX_LOADED_MODELS = 2

# Set up logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("JSON_LOGGING", "true").lower() == "true",
    log_file="server.log"
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LocalLLM Chat Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Disable Flash Attention warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global cache for loaded models and tokenizers
loaded_models: Dict[str, Any] = {}
loaded_tokenizers: Dict[str, Any] = {}

class ChatMessage(BaseModel):
    """
    Represents a single message in the chat conversation.
    """
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """
    Represents a chat request with conversation history and parameters.
    """
    model_config = ConfigDict(protected_namespaces=())
    
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model_id: str = Field(default=DEFAULT_MODEL, description="ID of the model to use")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens to generate")

def check_system_resources() -> Dict[str, Any]:
    """
    Check system resources and verify if there's enough capacity.
    
    Returns:
        Dict containing resource information
    
    Raises:
        RuntimeError: If system resources are insufficient
    """
    resources = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_free_gb": shutil.disk_usage("/").free / (1024**3)
    }
    
    if torch.cuda.is_available():
        resources.update({
            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
        })
    
    # Log resource status
    logger.info("System resources", extra={"metrics": resources})
    
    # Check for critical resource issues
    if resources["memory_percent"] > 90:
        raise RuntimeError("System memory usage too high")
    if resources["disk_free_gb"] < 10:
        raise RuntimeError("Insufficient disk space")
    
    return resources

def load_model(model_id: str, background_tasks: BackgroundTasks) -> Tuple[Any, Any]:
    """
    Load a model and its tokenizer if not already loaded.
    
    Args:
        model_id: ID of the model to load
        background_tasks: FastAPI background tasks for cleanup
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ValueError: If model_id is invalid
        RuntimeError: If system resources are insufficient
    """
    try:
        # Validate model ID
        if not validate_model_id(model_id):
            raise ValueError(f"Invalid model ID: {model_id}")
            
        # Get model config
        config = get_model_config(model_id)
        
        # Check system resources
        check_system_resources()
        
        # Return cached model if available
        if model_id in loaded_models:
            return loaded_models[model_id], loaded_tokenizers[model_id]
            
        # Clean up unused models
        if len(loaded_models) >= MAX_LOADED_MODELS:
            cleanup_unused_models()
            background_tasks.add_task(cleanup_unused_models)
            
        # Load tokenizer if not cached
        if model_id not in loaded_tokenizers:
            logger.info("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_id"],
                token=os.getenv("HUGGING_FACE_HUB_TOKEN")
            )
            
            # Ensure tokenizer has required special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            loaded_tokenizers[model_id] = tokenizer
            
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model
        logger.info("Loading model")
        start_time = time.time()
        
        # Get base config first
        base_config = AutoConfig.from_pretrained(
            config["model_id"],
            token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
            use_safetensors=True,
            trust_remote_code=config["trust_remote_code"]
        )
        
        # Load model with config
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=config["trust_remote_code"],
            device_map="auto",
            config=base_config,
            use_safetensors=True
        )
        
        load_time = time.time() - start_time
        loaded_models[model_id] = model
        
        # Log model load time
        logger.info(
            f"Model {model_id} loaded",
            extra={
                "load_time": load_time,
                "model_id": model_id
            }
        )
        
        return model, loaded_tokenizers[model_id]
        
    except Exception as e:
        logger.error(
            f"Failed to load model {model_id}",
            extra={"error": str(e)},
            exc_info=True
        )
        raise

def cleanup_unused_models():
    """
    Remove least recently used models to free up memory.
    """
    if len(loaded_models) <= 1:
        return
        
    # Keep the default model and the most recently used model
    models_to_keep = {DEFAULT_MODEL}
    if loaded_models and DEFAULT_MODEL not in loaded_models:
        models_to_keep.add(next(iter(loaded_models)))
        
    # Unload other models
    for model_id in list(loaded_models.keys()):
        if model_id not in models_to_keep:
            logger.info(f"Unloading model {model_id}")
            del loaded_models[model_id]
            if model_id in loaded_tokenizers:
                del loaded_tokenizers[model_id]
            torch.cuda.empty_cache()

async def generate_response(
    messages: List[ChatMessage],
    model: Any,
    tokenizer: Any,
    model_id: str,
    max_new_tokens: Optional[int] = None
) -> str:
    """Generate response from model"""
    config = get_model_config(model_id)
    max_new_tokens = max_new_tokens or config["max_new_tokens"]
    
    # Convert messages to dict format
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    # Format input text
    input_text = format_chat_messages(messages_dict)
    
    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=config["temperature"],
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean up response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Remove any system or human prefixes from response
    response = response.replace("### System:", "").replace("### Human:", "").replace("### Assistant:", "").strip()
    
    return response

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint providing system status"""
    try:
        resources = check_system_resources()
        return {
            "status": "healthy",
            "loaded_models": list(loaded_models.keys()),
            "total_models": len(MODELS_CONFIG),
            "resources": resources
        }
    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
def list_models() -> Dict[str, Any]:
    """List available models and their configurations"""
    return {
        "models": MODELS_CONFIG,
        "loaded_models": list(loaded_models.keys()),
        "default_model": DEFAULT_MODEL
    }

@app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Chat endpoint handling message generation"""
    try:
        # Load model
        model, tokenizer = load_model(request.model_id, background_tasks)
        
        # Generate response
        response = await generate_response(
            messages=request.messages,
            model=model,
            tokenizer=tokenizer,
            model_id=request.model_id,
            max_new_tokens=request.max_tokens
        )
        
        return {"response": response}
        
    except Exception as e:
        logger.error(
            "Chat request failed",
            extra={"error": str(e), "model_id": request.model_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)