import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def setup_models():
    """
    Copies the locally vendored MobileSAM weights to the app/models directory.
    No internet connection is required.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "app", "models")
    checkpoint_path = os.path.join(models_dir, "mobile_sam.pt")
    vendor_path = os.path.join(base_dir, "vendor", "MobileSAM", "weights", "mobile_sam.pt")
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Weights already exist at: {checkpoint_path}")
        return
        
    logger.info(f"Copying local vendored MobileSAM weights to {checkpoint_path}...")
    os.makedirs(models_dir, exist_ok=True)
    
    if os.path.exists(vendor_path):
        shutil.copy2(vendor_path, checkpoint_path)
        logger.info("Local weight copy successful. No download required!")
    else:
        logger.error(f"Vendored weights not found at {vendor_path}")
        raise FileNotFoundError(f"Missing vendored weights at {vendor_path}")

if __name__ == "__main__":
    setup_models()
