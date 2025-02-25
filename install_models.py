"""
Model Installation Module

This module handles the downloading and verification of models from HuggingFace.
It includes features for:
- Checking available disk space
- Downloading models with progress tracking
- MD5 checksum verification
- Retry logic for failed downloads
- Structured logging of installation progress
- Handling gated model access requests
"""

import os
import sys
import hashlib
import requests
import shutil
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import time
import json

from huggingface_hub import (
    HfApi,
    HfFolder,
    Repository,
    create_repo,
    get_token,
    hf_hub_download,
    login,
)

from config.models import (
    MODELS_CONFIG,
    get_model_config,
    get_total_required_space,
    validate_model_id
)
from utils.logging import setup_logging
from utils.model_access import check_model_access, request_model_access, wait_for_model_access

# Set up logging
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_output=os.getenv("JSON_LOGGING", "true").lower() == "true",
    log_file="install.log"
)

# Environment configuration
REQUIRE_CHECKSUMS = os.getenv("REQUIRE_CHECKSUMS", "false").lower() == "true"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
CACHE_DIR = Path(os.getenv("HUGGINGFACE_HUB_CACHE", DEFAULT_CACHE_DIR))

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
            "files": files,
            "total_size": sum(f.size for f in files if f.size is not None)
        }
        
    except Exception as e:
        logger.error(
            "Failed to get model files",
            extra={"error": str(e), "model_id": model_id},
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
    # Get Hugging Face token
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                token = f.read().strip()
        except:
            pass
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    
    # Create parent directories if they don't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            
            # Get content length, default to 0 if not provided
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=desc
            ) as progress_bar:
                with open(dest_path, 'wb') as f:
                    downloaded_size = 0
                    for data in response.iter_content(block_size):
                        downloaded_size += len(data)
                        progress_bar.update(len(data))
                        f.write(data)
                        
            # Only check size if content-length was provided
            if total_size != 0 and downloaded_size != total_size:
                raise RuntimeError("Downloaded size does not match expected size")
                
            # Verify file exists and has content
            if not dest_path.exists() or dest_path.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty or does not exist")
                
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

def check_model_files(model_key: str) -> bool:
    """
    Check if all model files are present.
    
    Args:
        model_key: Key in MODELS_CONFIG for the model to check
        
    Returns:
        Boolean indicating if all files are present
    """
    try:
        config = get_model_config(model_key)
        files = get_model_files(config["model_id"])
        
        # Check each file exists
        for file in files["files"]:
            file_path = CACHE_DIR / file.rfilename
            if not file_path.exists():
                return False
                
        return True
        
    except Exception as e:
        logger.warning(
            f"Failed to check model files",
            extra={
                "error": str(e),
                "model_key": model_key
            }
        )
        return False

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
    
    # Check disk space at cache directory
    if not check_disk_space(config["size_gb"], str(CACHE_DIR)):
        logger.error(
            "Insufficient disk space",
            extra={"required_gb": config["size_gb"], "cache_dir": str(CACHE_DIR)}
        )
        return False
    
    try:
        # Check and request model access if needed
        if not check_model_access(config["model_id"]):
            logger.info(f"Model {config['name']} requires access request")
            
            # Try to request access
            request_result = request_model_access(config["model_id"])
            if not request_result:
                logger.error(
                    f"Failed to request access to {config['name']}",
                    extra={
                        "model_id": model_id,
                        "error_type": "request_failed"
                    }
                )
                return False
                
            # Wait for access to be granted
            wait_result = wait_for_model_access(config["model_id"])
            if not wait_result:
                logger.error(
                    f"Access not granted to {config['name']}",
                    extra={
                        "model_id": model_id,
                        "error_type": "access_timeout"
                    }
                )
                return False
            
            # Double check access was actually granted
            if not check_model_access(config["model_id"]):
                logger.error(
                    f"Access verification failed for {config['name']}",
                    extra={
                        "model_id": model_id,
                        "error_type": "access_verification_failed"
                    }
                )
                return False
        
        # Get model files
        files = get_model_files(config["model_id"])
        
        # Create cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get checksums if available
        checksums = {}
        if config.get("checksum_url"):
            try:
                response = requests.get(config["checksum_url"])
                response.raise_for_status()
                checksum_text = response.text
                checksums = {
                    line.split()[1]: line.split()[0]
                    for line in checksum_text.splitlines()
                    if len(line.split()) >= 2
                }
            except Exception as e:
                msg = "Failed to fetch checksums"
                if REQUIRE_CHECKSUMS:
                    logger.error(
                        msg,
                        extra={"error": str(e), "url": config["checksum_url"]},
                        exc_info=True
                    )
                    return False
                else:
                    logger.warning(
                        msg,
                        extra={"error": str(e), "url": config["checksum_url"]}
                    )
        
        # Download and verify each file
        success = True
        for file in files["files"]:
            file_path = CACHE_DIR / file.rfilename
            expected_size = file.size if file.size is not None else None
            expected_md5 = checksums.get(file.rfilename)
            
            # Check if file exists and verify its integrity
            if file_path.exists() and not force:
                file_valid = True
                
                # Check file size if we know the expected size
                if expected_size is not None:
                    actual_size = file_path.stat().st_size
                    if actual_size != expected_size:
                        logger.warning(
                            f"Size mismatch for {file.rfilename}",
                            extra={
                                "expected_size": expected_size,
                                "actual_size": actual_size
                            }
                        )
                        file_valid = False
                
                # Check MD5 if available
                if expected_md5 and file_valid:
                    if not verify_checksum(file_path, expected_md5):
                        logger.warning(
                            f"Checksum mismatch for {file.rfilename}",
                            extra={
                                "file": str(file_path),
                                "expected_md5": expected_md5
                            }
                        )
                        file_valid = False
                
                if file_valid:
                    logger.info(f"File already exists and is valid: {file.rfilename}")
                    continue
                else:
                    logger.warning(f"File exists but is invalid, re-downloading: {file.rfilename}")
            
            # Download file
            success &= download_file(
                url=f"https://huggingface.co/{config['model_id']}/resolve/main/{file.rfilename}",
                dest_path=file_path,
                desc=f"Downloading {file.rfilename}"
            )
            
            if not success:
                logger.error(f"Failed to download {file.rfilename}")
                return False
            
            # Verify size
            if expected_size is not None:
                actual_size = file_path.stat().st_size
                if actual_size != expected_size:
                    logger.error(
                        f"Size verification failed for {file.rfilename}",
                        extra={
                            "expected_size": expected_size,
                            "actual_size": actual_size
                        }
                    )
                    return False
            
            # Verify checksum
            if expected_md5:
                if not verify_checksum(file_path, expected_md5):
                    logger.error(
                        f"Checksum verification failed for {file.rfilename}",
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Install models from HuggingFace")
    parser.add_argument("--force", action="store_true", help="Force re-download of models")
    parser.add_argument("--model", help="Specific model to download. If not specified, downloads all models.")
    args = parser.parse_args()

    # Unset HF_TOKEN to ensure HUGGING_FACE_HUB_TOKEN is used
    if "HF_TOKEN" in os.environ:
        logger.warning("HF_TOKEN is set, unsetting it to use HUGGING_FACE_HUB_TOKEN instead")
        del os.environ["HF_TOKEN"]

    # Check for HuggingFace token
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                token = f.read().strip()
        except:
            pass
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
    if not check_disk_space(total_gb, str(CACHE_DIR)):
        logger.error(
            "Insufficient disk space for all models",
            extra={"required_gb": total_gb, "cache_dir": str(CACHE_DIR)}
        )
        sys.exit(1)
    
    # Track overall success
    success = True
    
    # Download specific model or all models
    if args.model:
        if not validate_model_id(args.model):
            logger.error(f"Invalid model ID: {args.model}")
            sys.exit(1)
        success = download_model(args.model, force=args.force)
    else:
        # Download all models
        for model_id in MODELS_CONFIG:
            model_success = download_model(model_id, force=args.force)
            success &= model_success
            if not model_success:
                logger.error(f"Failed to download {model_id}")
    
    if not success:
        sys.exit(1)
        
    logger.info("All models downloaded successfully")

if __name__ == "__main__":
    main()
