from pathlib import Path
import numpy as np
from app.models.factory import SegmentationModelFactory
from app.models.mobile_sam_engine import MobileSAMEngine
from app.storage.factory import StorageProviderFactory
from app.config import settings

def test_model_factory_get_engine():
    sam_engine = SegmentationModelFactory.get_engine("mobile_sam")
    assert isinstance(sam_engine, MobileSAMEngine)

    # Default and unknown providers resolve to MobileSAM (MobileSAM-only service)
    assert isinstance(SegmentationModelFactory.get_engine(), MobileSAMEngine)
    assert isinstance(SegmentationModelFactory.get_engine("some_unknown_provider"), MobileSAMEngine)

def test_mobile_sam_engine_prediction():
    import cv2
    sample_path = str(Path(__file__).parent.parent / "storage" / "sample_hat.png")
    image = cv2.imread(sample_path)

    engine = SegmentationModelFactory.get_engine("mobile_sam")
    mask, confidence = engine.predict_mask(image, point=(300, 200), cache_key=sample_path)

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (600, 600)
    assert confidence > 0.5
    assert mask.any()
