"""
Model Installation Module

This module handles the downloading and verification of models from HuggingFace.
It includes features for:
- Checking available disk space
- Downloading models with progress tracking
- MD5 checksum verification
- Retry logic for failed downloads
- Structured logging of installation progress
"""

import os
import sys
import hashlib
import requests
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import time
import json

from huggingface_hub import (
    HfApi,
    hf_hub_download,
    login,
    ModelFilter,
    CommitInfo
)

from config.models import (
    MODELS_CONFIG,
    get_model_config,
    get_total_required_space,
    validate_model_id
)
from utils.logging import setup_logging

# Set up logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("JSON_LOGGING", "true").lower() == "true",
    log_file="install.log"
)

import logging
logger = logging.getLogger(__name__)

def check_disk_space(required_gb: float, path: str = ".") -> bool:
    """
    Check if there's enough disk space available.
    
    Args:
        required_gb: Required space in gigabytes
        path: Path to check space on
        
    Returns:
        Boolean indicating if there's enough space
    """
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    
    # Add 20% buffer for safety
    required_with_buffer = required_gb * 1.2
    
    logger.info(
        "Checking disk space",
        extra={
            "metrics": {
                "required_gb": required_gb,
                "free_gb": free_gb,
                "buffer_gb": required_with_buffer - required_gb
            }
        }
    )
    
    return free_gb >= required_with_buffer

def get_model_files(model_id: str) -> Dict[str, Any]:
    """
    Get information about model files from HuggingFace.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Dict containing file information
        
    Raises:
        ValueError: If model files cannot be retrieved
    """
    try:
        api = HfApi()
        files = api.model_info(
            model_id,
            token=os.getenv("HUGGING_FACE_HUB_TOKEN")
        ).siblings
        
        return {
            file.rfilename: {
                "size": file.size,
                "blob_id": file.blob_id
            }
            for file in files
        }
    except Exception as e:
        logger.error(
            f"Failed to get model files for {model_id}",
            extra={"error": str(e)},
            exc_info=True
        )
        raise ValueError(f"Could not get model files: {str(e)}")

def download_file(
    url: str,
    dest_path: Path,
    desc: str,
    max_retries: int = 3
) -> bool:
    """
    Download a file with progress tracking and retries.
    
    Args:
        url: URL to download from
        dest_path: Path to save file to
        desc: Description for progress bar
        max_retries: Maximum number of retry attempts
        
    Returns:
        Boolean indicating success
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=desc
            ) as progress_bar:
                with open(dest_path, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                        
            if total_size != 0 and progress_bar.n != total_size:
                raise RuntimeError("Downloaded size does not match expected size")
                
            return True
            
        except Exception as e:
            logger.warning(
                f"Download attempt {attempt + 1} failed",
                extra={
                    "error": str(e),
                    "url": url,
                    "attempt": attempt + 1
                }
            )
            if attempt + 1 == max_retries:
                logger.error(
                    "Download failed after all retries",
                    extra={"url": url},
                    exc_info=True
                )
                return False
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return False

def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """
    Verify file MD5 checksum.
    
    Args:
        file_path: Path to file to verify
        expected_md5: Expected MD5 hash
        
    Returns:
        Boolean indicating if checksum matches
    """
    md5_hash = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
            
    actual_md5 = md5_hash.hexdigest()
    
    logger.info(
        "Verifying checksum",
        extra={
            "file": str(file_path),
            "expected_md5": expected_md5,
            "actual_md5": actual_md5
        }
    )
    
    return actual_md5 == expected_md5

def download_model(model_id: str, force: bool = False) -> bool:
    """
    Download a model and verify its integrity.
    
    Args:
        model_id: ID of model to download
        force: Whether to force download even if files exist
        
    Returns:
        Boolean indicating success
    """
    if not validate_model_id(model_id):
        logger.error(f"Invalid model ID: {model_id}")
        return False
        
    config = get_model_config(model_id)
    logger.info(f"Preparing to download {config['name']}")
    
    # Check disk space
    if not check_disk_space(config["size_gb"]):
        logger.error(
            "Insufficient disk space",
            extra={"required_gb": config["size_gb"]}
        )
        return False
    
    try:
        # Get model files
        files = get_model_files(config["model_id"])
        
        # Create cache directory
        cache_dir = Path(".cache/huggingface")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download and verify each file
        success = True
        for filename, info in files.items():
            file_path = cache_dir / filename
            
            if file_path.exists() and not force:
                logger.info(f"File already exists: {filename}")
                continue
                
            # Download file
            success &= download_file(
                url=f"https://huggingface.co/{config['model_id']}/resolve/main/{filename}",
                dest_path=file_path,
                desc=f"Downloading {filename}"
            )
            
            if not success:
                logger.error(f"Failed to download {filename}")
                return False
                
            # Verify checksum if available
            if config.get("checksum_url"):
                checksums = requests.get(config["checksum_url"]).text
                expected_md5 = next(
                    (line.split()[0] for line in checksums.splitlines()
                     if filename in line),
                    None
                )
                
                if expected_md5:
                    if not verify_checksum(file_path, expected_md5):
                        logger.error(
                            f"Checksum verification failed for {filename}",
                            extra={
                                "file": str(file_path),
                                "model_id": model_id
                            }
                        )
                        return False
                        
        logger.info(
            f"Successfully downloaded {config['name']}",
            extra={"model_id": model_id}
        )
        return True
        
    except Exception as e:
        logger.error(
            f"Error downloading {model_id}",
            extra={"error": str(e)},
            exc_info=True
        )
        return False

def main():
    """Main entry point for model installation."""
    # Check for HuggingFace token
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.error("HUGGING_FACE_HUB_TOKEN not set in environment")
        sys.exit(1)
        
    # Login to HuggingFace
    try:
        login(token=token)
    except Exception as e:
        logger.error(
            "Failed to login to HuggingFace",
            extra={"error": str(e)},
            exc_info=True
        )
        sys.exit(1)
    
    # Calculate total required space
    total_gb = get_total_required_space()
    
    # Check total disk space
    if not check_disk_space(total_gb):
        logger.error(
            "Insufficient disk space for all models",
            extra={"required_gb": total_gb}
        )
        sys.exit(1)
    
    # Download each model
    success = True
    for model_id in MODELS_CONFIG:
        if not download_model(model_id):
            success = False
            logger.error(f"Failed to download {model_id}")
    
    if not success:
        logger.error("Some models failed to download")
        sys.exit(1)
        
    logger.info("All models downloaded successfully")

if __name__ == "__main__":
    main()
