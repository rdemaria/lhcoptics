# tests/conftest.py
import pytest
import requests
import tempfile
import os


@pytest.fixture(scope="session")
def lhc_json_file():
    """Downloads a file from a given URL and returns the local file path."""

    url = "https://gitlab.cern.ch/acc-models/acc-models-lhc/-/raw/2025/xsuite/lhc.json?ref_type=heads"  # Replace with actual URL
    filename = os.path.basename(url)

    # Use a temporary directory
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    # Download only if not already present
    if not os.path.exists(file_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    return file_path
