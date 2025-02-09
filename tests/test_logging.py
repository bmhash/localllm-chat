"""
Tests for the logging module.
"""

import pytest
import json
import logging
import time
from unittest.mock import patch, MagicMock
from utils.logging import (
    JSONFormatter,
    PerformanceMetrics,
    setup_logging
)

def test_json_formatter():
    """Test JSON formatter output."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None
    )
    
    output = formatter.format(record)
    data = json.loads(output)
    
    assert isinstance(data, dict)
    assert data["level"] == "INFO"
    assert data["message"] == "Test message"
    assert "timestamp" in data

def test_setup_logging():
    """Test logging setup."""
    setup_logging(level=logging.INFO, json_output=True)
    root_logger = logging.getLogger()
    
    assert root_logger.level == logging.INFO
    assert len(root_logger.handlers) > 0
    
    # Test performance logger
    perf_logger = logging.getLogger("performance")
    assert len(perf_logger.handlers) > 0

def test_performance_metrics():
    """Test performance metrics tracking."""
    metrics = PerformanceMetrics()
    
    # Test resource usage logging
    resource_metrics = metrics.log_resource_usage()
    assert isinstance(resource_metrics, dict)
    assert "cpu_percent" in resource_metrics
    assert "ram_usage_mb" in resource_metrics
    
    # Test inference time logging
    inference_metrics = metrics.log_inference_time(
        model_id="test-model",
        start_time=time.time() - 1,  # 1 second ago
        input_tokens=10,
        output_tokens=20
    )
    assert isinstance(inference_metrics, dict)
    assert inference_metrics["model_id"] == "test-model"
    assert inference_metrics["input_tokens"] == 10
    assert inference_metrics["output_tokens"] == 20
    assert inference_metrics["duration_seconds"] > 0
