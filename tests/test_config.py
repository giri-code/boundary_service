from app.config import settings

def test_config_defaults():
    assert settings.APP_NAME == "Object Boundary Detection Service"
    assert settings.VERSION == "1.0.0"
    assert settings.DEFAULT_POLYGON_TOLERANCE == 0.005
    assert settings.MAX_IMAGE_FILE_SIZE_MB > 0
    assert settings.MAX_IMAGE_PIXELS > 0
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000
    assert settings.MIN_HOLE_AREA_PIXELS == 10
    assert ".jpg" in settings.ALLOWED_IMAGE_EXTENSIONS
    assert ".png" in settings.ALLOWED_IMAGE_EXTENSIONS

