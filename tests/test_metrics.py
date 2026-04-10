"""Tests for metrics module."""

import tempfile
from pathlib import Path

from longtext_pipeline.utils.metrics import (
    llm_latency_seconds,
    llm_requests_total,
    rate_limit_hits_total,
    retry_attempts_total,
    retry_delay_seconds,
    write_metrics_to_file,
)


def test_metrics_are_defined():
    """Verify all 5 metrics are properly defined."""
    assert retry_attempts_total is not None
    assert retry_delay_seconds is not None
    assert rate_limit_hits_total is not None
    assert llm_requests_total is not None
    assert llm_latency_seconds is not None


def test_counter_metrics_increment():
    """Test that counter metrics can be incremented."""
    # Reset counters before test by setting to 0
    retry_attempts_total.labels(stage="test", error_type="timeout")._value.set(0)
    rate_limit_hits_total.labels(stage="test")._value.set(0)
    llm_requests_total.labels(stage="test", status="success")._value.set(0)

    # Increment counters
    retry_attempts_total.labels(stage="test", error_type="timeout").inc()
    rate_limit_hits_total.labels(stage="test").inc(2)
    llm_requests_total.labels(stage="test", status="success").inc()

    # Verify increments
    assert (
        retry_attempts_total.labels(stage="test", error_type="timeout")._value.get()
        == 1
    )
    assert rate_limit_hits_total.labels(stage="test")._value.get() == 2
    assert llm_requests_total.labels(stage="test", status="success")._value.get() == 1


def test_histogram_metrics_observe():
    """Test that histogram metrics can record observations."""
    # Reset histogram
    retry_delay_seconds.labels(stage="test_observe").observe(0)

    # Observe a value
    retry_delay_seconds.labels(stage="test_observe").observe(5.0)

    # Observe another value
    llm_latency_seconds.labels(stage="test_observe").observe(2.5)

    # If we get here without errors, the observe function works
    assert True


def test_write_metrics_to_file():
    """Test writing metrics to Prometheus text format file."""
    # Create some sample metrics
    retry_attempts_total.labels(stage="summarize", error_type="timeout").inc(3)
    llm_latency_seconds.labels(stage="summarize").observe(1.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)

        # Write metrics to file
        write_metrics_to_file(output_path)

        # Verify file was created
        metrics_file = output_path / ".longtext" / "metrics.prom"
        assert metrics_file.exists()

        # Verify file contains Prometheus format
        content = metrics_file.read_text()
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "retry_attempts_total" in content
        assert "llm_latency_seconds" in content
