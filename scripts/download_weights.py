import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Sanity floor: a truncated download is never this small (real file ~40MB).
MIN_WEIGHT_BYTES = 1_000_000


def setup_models():
    """
    Ensure MobileSAM weights exist where app/config.py looks:
    <repo>/models/mobile_sam.pt (BASE_DIR / "models").

    Source of truth is the vendored copy at
    <repo>/vendor/MobileSAM/weights/mobile_sam.pt — no internet required.
    Idempotent: skips the copy when a non-empty checkpoint already exists.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    checkpoint_path = os.path.join(models_dir, "mobile_sam.pt")
    vendor_path = os.path.join(base_dir, "vendor", "MobileSAM", "weights", "mobile_sam.pt")

    if os.path.exists(checkpoint_path):
        size = os.path.getsize(checkpoint_path)
        if size >= MIN_WEIGHT_BYTES:
            logger.info(f"Weights already exist at: {checkpoint_path} ({size} bytes)")
            return
        raise IOError(
            f"Checkpoint at {checkpoint_path} is truncated ({size} bytes). "
            "Delete it and re-run this script."
        )

    logger.info(f"Copying local vendored MobileSAM weights to {checkpoint_path}...")
    os.makedirs(models_dir, exist_ok=True)

    if os.path.exists(vendor_path):
        if os.path.getsize(vendor_path) < MIN_WEIGHT_BYTES:
            raise IOError(f"Vendored weights truncated at {vendor_path}")
        shutil.copy2(vendor_path, checkpoint_path)
        logger.info("Local weight copy successful. No download required!")
    else:
        logger.error(f"Vendored weights not found at {vendor_path}")
        raise FileNotFoundError(f"Missing vendored weights at {vendor_path}")

if __name__ == "__main__":
    setup_models()
