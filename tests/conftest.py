import cv2
from pathlib import Path
import pytest
from unittest.mock import patch

SAMPLE_HAT_PATH = Path(__file__).parent.parent / "storage" / "sample_hat.png"

@pytest.fixture(autouse=True)
def mock_storage_read_image():
    """Auto-mock StorageProviderFactory.read_image for unit tests."""
    sample_img = cv2.imread(str(SAMPLE_HAT_PATH))

    def fake_read_image(path_or_uri: str):
        if "passwd" in path_or_uri or "traversal" in path_or_uri:
            raise PermissionError(f"Access denied: '{path_or_uri}' contains invalid directory traversal.")
        if "non_existent" in path_or_uri or "non/existent" in path_or_uri:
            raise FileNotFoundError(f"Image not found: {path_or_uri}")
        return sample_img

    with patch("app.storage.factory.StorageProviderFactory.read_image", side_effect=fake_read_image):
        yield
