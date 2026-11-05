import io
import zipfile
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session")
def glitchsim_regression_data_archive() -> str:
    """
    Download link that can be used to fetch regression data for the `glitchsim` scripts.

    Deliberately hardcoded. Last updated 2026-11-05.
    """
    return "https://github.com/NSs-FLF-RHUL/deepSKA/releases/download/glitchsim-regression-outputs/graber-cumming-anderson-2018-recreation.zip"


@pytest.fixture(scope="session")
def regression_data(
    tmp_path_factory: pytest.TempPathFactory,
    glitchsim_regression_data_archive: str,
    subdirectory: str = "regression_data",
) -> Path:
    """
    Fetch reference data for regression tests.

    This will fetch the reference data (as a `.zip` archive) from the
    pre-defined location, and extract it to the test session's `tmp_path`.

    This fixture only runs once per session, so data does not have to be
    continually re-downloaded after each test. Conversely, this means that
    the created folder and files within _must not_ be edited during tests,
    and should be read-only.

    This fixture returns the `Path` to which the data was extracted.
    """
    extract_to = tmp_path_factory.mktemp(subdirectory)

    response = requests.get(glitchsim_regression_data_archive, timeout=30.0)

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    archive.extractall(extract_to)

    return extract_to
