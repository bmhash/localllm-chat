"""
Logging Configuration Module

This module provides a centralized logging configuration for the application.
It sets up structured logging with various handlers and formatters for different
logging requirements.

Features:
- Structured JSON logging for machine parsing
- Console output for development
- File output for production
- Performance metrics tracking
- Resource usage monitoring
- Error tracking with stack traces
"""

import logging
import json
import os
import sys
import time
import psutil
import torch
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format for easy parsing.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": str(record.exc_info[0].__name__),
                "message": str(record.exc_info[1]),
                "stack_trace": self.formatException(record.exc_info)
            }
            
        return json.dumps(log_data)

class PerformanceMetrics:
    """
    Track and log performance metrics including:
    - GPU memory usage
    - CPU usage
    - RAM usage
    - Inference times
    - Token counts
    """
    def __init__(self):
        self.process = psutil.Process()
        self.logger = logging.getLogger("performance")
        
    def log_resource_usage(self) -> Dict[str, Any]:
        """Log system resource usage."""
        metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "ram_percent": psutil.virtual_memory().percent,
        }

        # Only add GPU metrics if CUDA is available and pynvml is installed
        try:
            if torch.cuda.is_available():
                metrics["gpu_utilization"] = torch.cuda.utilization()
        except (ImportError, AttributeError):
            pass  # GPU metrics not available

        self.logger.info("Resource usage metrics", extra={"metrics": metrics})
        return metrics
        
    def log_inference_time(self, model_id: str, start_time: float, 
                          input_tokens: int, output_tokens: int):
        """Log model inference performance."""
        end_time = time.time()
        duration = end_time - start_time
        
        metrics = {
            "model_id": model_id,
            "duration_seconds": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": output_tokens / duration if duration > 0 else 0
        }
        
        self.logger.info("Inference metrics", extra={"metrics": metrics})
        return metrics

def setup_logging(
    level: int = logging.INFO,
    json_output: bool = True,
    log_file: Optional[str] = None
) -> None:
    """
    Set up application-wide logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        json_output: Whether to use JSON formatting (default: True)
        log_file: Optional file path for logging output
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Create formatters
    json_formatter = JSONFormatter()
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter if not json_output else json_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(LOGS_DIR / log_file)
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)
    
    # Create performance logger
    perf_logger = logging.getLogger("performance")
    perf_handler = logging.FileHandler(LOGS_DIR / "performance.json")
    perf_handler.setFormatter(json_formatter)
    perf_logger.addHandler(perf_handler)
    
    # Create error logger
    error_logger = logging.getLogger("errors")
    error_handler = logging.FileHandler(LOGS_DIR / "errors.json")
    error_handler.setFormatter(json_formatter)
    error_logger.addHandler(error_handler)

# Initialize performance metrics
metrics = PerformanceMetrics()
