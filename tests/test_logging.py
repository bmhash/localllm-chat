"""
Tests for the logging module.
"""

import pytest
import logging
import json
import os
from pathlib import Path
from utils.logging import (
    setup_logging,
    JSONFormatter,
    PerformanceMetrics
)

def test_json_formatter():
    """Test that JSONFormatter formats logs correctly."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test")
    
    # Create a log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    # Format the record
    formatted = formatter.format(record)
    
    # Parse the JSON output
    log_data = json.loads(formatted)
    
    # Check required fields
    assert "timestamp" in log_data
    assert log_data["level"] == "INFO"
    assert log_data["message"] == "Test message"
    assert log_data["module"] == "test"
    assert log_data["line"] == 1

def test_setup_logging(tmp_path):
    """Test logging setup."""
    log_file = tmp_path / "test.log"
    
    # Setup logging
    setup_logging(
        level=logging.INFO,
        json_output=True,
        log_file=str(log_file)
    )
    
    # Get logger and log a message
    logger = logging.getLogger("test")
    test_message = "Test log message"
    logger.info(test_message)
    
    # Check that file was created
    assert log_file.exists()
    
    # Read log file and check content
    with open(log_file) as f:
        log_data = json.loads(f.readline())
        assert log_data["message"] == test_message

def test_performance_metrics():
    """Test PerformanceMetrics class."""
    metrics = PerformanceMetrics()
    
    # Test resource usage logging
    resource_metrics = metrics.log_resource_usage()
    assert "cpu_percent" in resource_metrics
    assert "ram_usage_mb" in resource_metrics
    assert "ram_percent" in resource_metrics
    
    # GPU metrics are optional and may not be available
    if "gpu_utilization" in resource_metrics:
        assert isinstance(resource_metrics["gpu_utilization"], (int, float))
    
    # Test inference time logging
    inference_metrics = metrics.log_inference_time(
        model_id="test-model",
        start_time=0,
        input_tokens=10,
        output_tokens=20
    )
    assert inference_metrics["model_id"] == "test-model"
    assert inference_metrics["input_tokens"] == 10
    assert inference_metrics["output_tokens"] == 20
    assert "duration_seconds" in inference_metrics
    assert "tokens_per_second" in inference_metrics
