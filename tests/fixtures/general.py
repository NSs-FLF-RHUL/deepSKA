from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Path to the tests/data directory."""
    return (Path(__file__) / ".." / "data").resolve()
