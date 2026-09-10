"""Unmocked storage tests: real factory + real S3 provider, fake boto3 transport.

tests/conftest.py auto-mocks StorageProviderFactory.read_image for speed and
fakes traversal rejection by substring match — so test_security.py's traversal
test never touches the real providers. These tests opt out via the
`real_s3_stack` fixture and prove the genuine routing/validation behavior.
"""

import cv2
import pytest
import httpx
from botocore.exceptions import ClientError

from app.main import app
from app.config import settings
from app.storage.s3_provider import S3StorageProvider
from tests.conftest import SAMPLE_HAT_PATH

SAMPLE_BYTES = SAMPLE_HAT_PATH.read_bytes()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_parse_s3_uri_unit():
    provider = S3StorageProvider(bucket_name="b")
    assert provider._parse_s3_uri("s3://b/k.png") == ("b", "k.png")
    assert provider._parse_s3_uri("s3://b/a/b/c.png") == ("b", "a/b/c.png")
    assert provider._parse_s3_uri("bare-key.png") == ("b", "bare-key.png")
    assert provider._parse_s3_uri("s3://") == ("", "")


def test_absolute_local_path_never_reads_disk(real_s3_stack, tmp_path):
    """A valid PNG sitting on local disk must NOT be read: absolute filesystem
    paths are treated as opaque S3 keys, so /etc/passwd-style reads are impossible."""
    provider, fake = real_s3_stack
    local_file = tmp_path / "secret.png"
    local_file.write_bytes(SAMPLE_BYTES)  # decodable, would succeed if read locally

    with pytest.raises(ClientError):
        provider.read_image(str(local_file))

    assert fake.head_calls == [("test-bucket", str(local_file))]
    assert fake.get_calls == []


def test_traversal_key_stays_opaque_s3_key(real_s3_stack):
    """`..` segments are passed to S3 verbatim — no local path resolution occurs."""
    provider, fake = real_s3_stack
    with pytest.raises(ClientError):
        provider.read_image("s3://test-bucket/../../etc/passwd")
    assert fake.head_calls == [("test-bucket", "../../etc/passwd")]


def test_oversize_object_rejected_before_fetch(real_s3_stack):
    provider, fake = real_s3_stack
    oversize = settings.MAX_IMAGE_FILE_SIZE_MB * 1024 * 1024 + 1
    fake.length_overrides[("test-bucket", "big.png")] = oversize
    with pytest.raises(ValueError, match="exceeds"):
        provider.read_image("s3://test-bucket/big.png")
    assert fake.get_calls == []


def test_undecodable_bytes_rejected(real_s3_stack):
    provider, fake = real_s3_stack
    fake.blobs[("test-bucket", "broken.png")] = b"not-an-image-at-all"
    with pytest.raises(ValueError, match="Failed to decode"):
        provider.read_image("s3://test-bucket/broken.png")


def test_valid_s3_object_decodes(real_s3_stack):
    provider, fake = real_s3_stack
    fake.blobs[("test-bucket", "ok.png")] = SAMPLE_BYTES
    image = provider.read_image("s3://test-bucket/ok.png")
    expected = cv2.imread(str(SAMPLE_HAT_PATH))
    assert image.shape == expected.shape


@pytest.mark.asyncio
async def test_api_empty_s3_uri_returns_400(real_s3_stack):
    """End-to-end through the real factory: invalid URI → 400, no S3 call made."""
    _, fake = real_s3_stack
    payload = {"image_path": "s3://", "x": 10, "y": 10}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/boundary",
            json=payload,
            headers={"X-Internal-Token": settings.INTERNAL_API_KEY},
        )
        assert response.status_code == 400
    assert fake.head_calls == []


@pytest.mark.asyncio
async def test_api_valid_s3_key_end_to_end(real_s3_stack):
    """End-to-end through real storage + real MobileSAM inference."""
    _, fake = real_s3_stack
    fake.blobs[("test-bucket", "hat.png")] = SAMPLE_BYTES
    payload = {"image_path": "s3://test-bucket/hat.png", "x": 300, "y": 200}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/boundary",
            json=payload,
            headers={"X-Internal-Token": settings.INTERNAL_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["outer_boundary"]) >= 3
