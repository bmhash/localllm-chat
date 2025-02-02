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
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import os
import psutil
import shutil
from pathlib import Path
import time
from config.models import (
    MODELS_CONFIG,
    DEFAULT_MODEL,
    get_model_config,
    validate_model_id
)
from utils.logging import setup_logging, metrics

# Set up logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("JSON_LOGGING", "true").lower() == "true",
    log_file="server.log"
)

import logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LocalLLM Chat Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disable Flash Attention warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global cache for loaded models and tokenizers
loaded_models: Dict[str, Any] = {}
loaded_tokenizers: Dict[str, Any] = {}

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)

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

def load_model(model_id: str, background_tasks: BackgroundTasks):
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
    if not validate_model_id(model_id):
        raise ValueError(f"Invalid model ID: {model_id}")
        
    if model_id not in loaded_models:
        logger.info(f"Loading model {model_id}")
        config = get_model_config(model_id)
        
        # Check system resources
        resources = check_system_resources()
        required_memory = config["size_gb"]
        if torch.cuda.is_available():
            available_memory = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                - resources["gpu_memory_allocated_gb"]
            )
            if available_memory < required_memory:
                raise RuntimeError(f"Insufficient GPU memory. Need {required_memory}GB, have {available_memory}GB available")
        
        # Load tokenizer if not already loaded
        if model_id not in loaded_tokenizers:
            logger.info("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_id"],
                trust_remote_code=config.get("trust_remote_code", False)
            )
            loaded_tokenizers[model_id] = tokenizer
            
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["load_in_4bit"],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=config.get("trust_remote_code", False)
        )
        loaded_models[model_id] = model
        
        # Log loading metrics
        load_time = time.time() - start_time
        logger.info(
            f"Model {model_id} loaded",
            extra={
                "metrics": {
                    "model_id": model_id,
                    "load_time_seconds": load_time,
                    "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3)
                }
            }
        )
        
        # Schedule cleanup of unused models
        background_tasks.add_task(cleanup_unused_models)
        
    return loaded_models[model_id], loaded_tokenizers[model_id]

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

def format_messages(messages: List[ChatMessage], model_id: str) -> List[Dict[str, str]]:
    """
    Format messages for the model while preserving context.
    
    Args:
        messages: List of chat messages
        model_id: ID of the model being used
        
    Returns:
        List of formatted messages
    """
    formatted_messages = []
    
    # Add system message
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

async def generate_response(
    messages: List[ChatMessage],
    model: Any,
    tokenizer: Any,
    model_id: str,
    max_new_tokens: Optional[int] = None
) -> str:
    """
    Generate response from model.
    
    Args:
        messages: List of chat messages
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_id: ID of the model being used
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated response text
    """
    config = get_model_config(model_id)
    max_new_tokens = max_new_tokens or config["max_new_tokens"]
    
    # Format conversation history
    formatted_messages = format_messages(messages, model_id)
    
    # Prepare input
    input_text = "\n".join([f"{m['role']}: {m['content']}" for m in formatted_messages])
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    
    # Track performance
    start_time = time.time()
    input_tokens = input_ids.shape[1]
    
    try:
        # Generate response
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=config["temperature"],
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode response
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text[len(input_text):].strip()
        
        # Log performance metrics
        output_tokens = output_ids.shape[1] - input_ids.shape[1]
        metrics.log_inference_time(
            model_id=model_id,
            start_time=start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Error generating response",
            extra={
                "error": str(e),
                "model_id": model_id,
                "input_tokens": input_tokens
            },
            exc_info=True
        )
        raise

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint providing system status.
    """
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
async def list_models() -> Dict[str, Any]:
    """
    List available models and their configurations.
    """
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
    """
    Chat endpoint handling message generation.
    """
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
        
        # Log resource usage
        metrics.log_resource_usage()
        
        return {"response": response}
        
    except Exception as e:
        logger.error(
            "Chat request failed",
            extra={"error": str(e), "model_id": request.model_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"Starting server with available models: {', '.join(MODELS_CONFIG.keys())}")
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Preload default model
    try:
        load_model(DEFAULT_MODEL, BackgroundTasks())
        logger.info(f"Preloaded default model: {DEFAULT_MODEL}")
    except Exception as e:
        logger.error(f"Failed to preload default model: {str(e)}", exc_info=True)