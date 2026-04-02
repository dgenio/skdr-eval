"""Pytest configuration for skdr-eval test suite."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")
