"""Pytest configuration and fixtures for longtext-pipeline tests.

This module provides shared fixtures and configuration for all tests.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def clear_prometheus_registry_at_start():
    """Clear Prometheus registry once at the start of the test session.

    This ensures a clean slate before any tests run. Prometheus metrics
    are global, so we clear them once to avoid 'Duplicated timeseries'
    errors when test modules import modules that define metrics.

    We do this at SESSION start, not after every test, because:
    1. Metrics are created at module import time (once per Python process)
    2. After clearing, imported modules don't re-create their metrics
    3. Tests that need metrics can still use them within their test file
    """
    from prometheus_client import REGISTRY

    # Collect all collectors to unregister (skip built-in ones)
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        # Skip built-in collectors
        collector_name = type(collector).__name__
        if collector_name in ("GCCollector", "PlatformCollector", "ProcessCollector"):
            continue
        try:
            REGISTRY.unregister(collector)
        except Exception:
            # Ignore errors
            pass
