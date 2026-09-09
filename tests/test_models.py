from pathlib import Path
import numpy as np
from app.models.factory import SegmentationModelFactory
from app.models.opencv_engine import OpenCVEngine
from app.models.mobile_sam_engine import MobileSAMEngine
from app.storage.factory import StorageProviderFactory
from app.config import settings

def test_model_factory_get_engine():
    cv_engine = SegmentationModelFactory.get_engine("opencv")
    assert isinstance(cv_engine, OpenCVEngine)

    sam_engine = SegmentationModelFactory.get_engine("mobile_sam")
    assert isinstance(sam_engine, MobileSAMEngine)

def test_opencv_engine_prediction():
    import cv2
    sample_path = str(Path(__file__).parent.parent / "storage" / "sample_hat.png")
    image = cv2.imread(sample_path)

    engine = OpenCVEngine()
    mask, confidence = engine.predict_mask(image, point=(300, 200), cache_key=sample_path)

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (600, 600)
    assert confidence > 0.5
    assert mask[200, 300] is np.bool_(True)
