import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def regression_data(
    tmp_path_factory: pytest.TempPathFactory, subdirectory: str = "regression_data"
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

    # Replace with fetch from DOI-registered site.
    fetch_from = Path(__file__).parent / "reference_data.zip"

    archive = zipfile.ZipFile(fetch_from, "r")
    archive.extractall(extract_to)

    return extract_to
