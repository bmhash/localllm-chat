"""
Model Access Management Module

This module handles access requests and verification for gated models on Hugging Face Hub.
"""

import time
from typing import Optional, Dict, Any
import requests
from huggingface_hub import HfApi, model_info
from huggingface_hub.utils import HfHubHTTPError
import logging

logger = logging.getLogger(__name__)

def check_model_access(model_id: str) -> bool:
    """
    Check if the current user has access to a model.
    
    Args:
        model_id: The Hugging Face model ID
        
    Returns:
        Boolean indicating if user has access
    """
    try:
        # Try to get model info - this will fail if we don't have access
        info = model_info(model_id)
        return True
    except HfHubHTTPError as e:
        if e.response.status_code == 401:  # Unauthorized
            return False
        elif e.response.status_code == 403:  # Forbidden/Rejected
            logger.error(f"Access to model {model_id} has been rejected")
            return False
        raise

def request_model_access(model_id: str) -> bool:
    """
    Request access to a gated model.
    
    Args:
        model_id: The Hugging Face model ID
        
    Returns:
        Boolean indicating if access was granted
    """
    api = HfApi()
    
    try:
        # First check if we already have access
        if check_model_access(model_id):
            logger.info(f"Already have access to {model_id}")
            return True
            
        # Try to request access
        logger.info(f"Requesting access to {model_id}")
        api.request_model_access(model_id)
        
        # Some models have automatic approval
        if check_model_access(model_id):
            logger.info(f"Access automatically granted to {model_id}")
            return True
            
        logger.info(f"Access request sent for {model_id}, waiting for manual approval")
        return False
        
    except Exception as e:
        logger.error(f"Failed to request access to {model_id}: {str(e)}")
        return False

def wait_for_model_access(model_id: str, timeout: int = 300, check_interval: int = 10) -> bool:
    """
    Wait for access to be granted to a model.
    
    Args:
        model_id: The Hugging Face model ID
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        
    Returns:
        Boolean indicating if access was granted
    """
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        if check_model_access(model_id):
            logger.info(f"Access granted to {model_id}")
            return True
            
        logger.info(f"Waiting for access to {model_id}...")
        time.sleep(check_interval)
    
    logger.error(f"Timed out waiting for access to {model_id}")
    return False
