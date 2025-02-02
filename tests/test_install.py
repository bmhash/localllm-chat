"""
Tests for the model installation module.
"""

import pytest
import os
from pathlib import Path
import requests
from install_models import (
    check_disk_space,
    verify_checksum,
    download_file
)

def test_check_disk_space(tmp_path):
    """Test disk space checking."""
    # Get actual free space
    free_space_gb = os.statvfs(tmp_path).f_frsize * \
                    os.statvfs(tmp_path).f_bavail / (1024**3)
    
    # Test with space we definitely have
    assert check_disk_space(free_space_gb * 0.5, tmp_path) is True
    
    # Test with space we definitely don't have
    assert check_disk_space(free_space_gb * 2, tmp_path) is False

def test_verify_checksum(tmp_path):
    """Test checksum verification."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_content = b"test content"
    test_file.write_bytes(test_content)
    
    # Calculate correct MD5
    import hashlib
    correct_md5 = hashlib.md5(test_content).hexdigest()
    
    # Test with correct checksum
    assert verify_checksum(test_file, correct_md5) is True
    
    # Test with incorrect checksum
    assert verify_checksum(test_file, "wrong_checksum") is False

def test_download_file(tmp_path):
    """Test file downloading with a small test file."""
    # Use a reliable test URL
    test_url = "https://httpbin.org/bytes/1024"
    dest_path = tmp_path / "downloaded.txt"
    
    success = download_file(
        url=test_url,
        dest_path=dest_path,
        desc="Testing download",
        max_retries=1
    )
    
    assert success is True
    assert dest_path.exists()
    assert dest_path.stat().st_size == 1024
