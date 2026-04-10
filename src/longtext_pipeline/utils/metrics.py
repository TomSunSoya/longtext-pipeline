"""Prometheus metrics module for longtext-pipeline observability.

This module defines Prometheus metrics for tracking pipeline performance,
retry behavior, rate limiting, and LLM request statistics.

Metrics are exported in Prometheus text format to .longtext/metrics.prom
for scraping by external monitoring systems.
"""

from pathlib import Path
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

# Use a custom registry to avoid conflicts with test reloads
REGISTRY = CollectorRegistry()


def _create_counter(name, description, labelnames):
    """Create a counter metric, safely handling re-registration."""
    try:
        return Counter(name, description, labelnames, registry=REGISTRY)
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Already registered - this can happen during test reloads
            # Return a dummy counter that won't raise errors on use
            return Counter(
                f"{name}_dummy", description, labelnames, registry=CollectorRegistry()
            )
        raise


def _create_histogram(name, description, labelnames, buckets):
    """Create a histogram metric, safely handling re-registration."""
    try:
        return Histogram(
            name, description, labelnames, buckets=buckets, registry=REGISTRY
        )
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Already registered
            return Histogram(
                f"{name}_dummy",
                description,
                labelnames,
                buckets=buckets,
                registry=CollectorRegistry(),
            )
        raise


# =============================================================================
# COUNTER METRICS
# =============================================================================

retry_attempts_total = _create_counter(
    "retry_attempts_total",
    "Total number of retry attempts per stage",
    ["stage", "error_type"],
)

rate_limit_hits_total = _create_counter(
    "rate_limit_hits_total",
    "Total number of rate limit hits per stage",
    ["stage"],
)

llm_requests_total = _create_counter(
    "llm_requests_total",
    "Total number of LLM requests per stage with status",
    ["stage", "status"],
)


# =============================================================================
# HISTOGRAM METRICS
# =============================================================================

retry_delay_seconds = _create_histogram(
    "retry_delay_seconds",
    "Distribution of retry delays per stage",
    ["stage"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

llm_latency_seconds = _create_histogram(
    "llm_latency_seconds",
    "Distribution of LLM request latency per stage",
    ["stage"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)


# =============================================================================
# METRICS EXPOSITION
# =============================================================================


def write_metrics_to_file(output_dir: str | Path) -> None:
    """Write all metrics to a Prometheus text format file.

    Args:
        output_dir: Directory to write metrics file (creates .longtext subdirectory)

    The output file (.longtext/metrics.prom) follows Prometheus text format:
    https://prometheus.io/docs/instrumenting/exposition_formats/#text-based-format

    Example output:
        # HELP retry_attempts_total Total number of retry attempts per stage
        # TYPE retry_attempts_total counter
        retry_attempts_total{stage="summarize",error_type="timeout"} 3.0
        # HELP llm_latency_seconds Distribution of LLM request latency per stage
        # TYPE llm_latency_seconds histogram
        llm_latency_seconds_bucket{stage="summarize",le="0.1"} 0.0
        llm_latency_seconds_bucket{stage="summarize",le="0.5"} 2.0
        llm_latency_seconds_sum{stage="summarize"} 1.234
        llm_latency_seconds_count{stage="summarize"} 5.0
    """
    output_path = Path(output_dir) / ".longtext"
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "metrics.prom"

    # Generate Prometheus text format using our custom registry
    metrics_text = generate_latest(REGISTRY).decode("utf-8")

    # Write to file
    metrics_file.write_text(metrics_text, encoding="utf-8")
