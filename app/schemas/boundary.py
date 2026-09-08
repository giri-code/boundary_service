from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    x: int = Field(
        ..., description="X pixel coordinate", json_schema_extra={"example": 300}
    )
    y: int = Field(
        ..., description="Y pixel coordinate", json_schema_extra={"example": 200}
    )


class BoundingBox(BaseModel):
    xmin: int = Field(
        ...,
        description="Minimum X coordinate of object bounding box",
        json_schema_extra={"example": 160},
    )
    ymin: int = Field(
        ...,
        description="Minimum Y coordinate of object bounding box",
        json_schema_extra={"example": 160},
    )
    xmax: int = Field(
        ...,
        description="Maximum X coordinate of object bounding box",
        json_schema_extra={"example": 441},
    )
    ymax: int = Field(
        ...,
        description="Maximum Y coordinate of object bounding box",
        json_schema_extra={"example": 286},
    )


class ImageDimensions(BaseModel):
    width: int = Field(
        ..., description="Image width in pixels", json_schema_extra={"example": 600}
    )
    height: int = Field(
        ..., description="Image height in pixels", json_schema_extra={"example": 600}
    )


class EncodeRequest(BaseModel):
    photo_id: str = Field(..., description="Unique ID of the photo")
    image_path: str = Field(
        ...,
        description="Local file path ('sample.jpg'), relative name, s3:// URI, gs:// URI, or HTTP presigned URL",
        json_schema_extra={"example": "sample_hat.png"},
    )


class BoundaryRequest(BaseModel):
    photo_id: Optional[str] = Field(
        None, description="Unique ID of the photo (if pre-encoded)"
    )
    image_path: Optional[str] = Field(
        None,
        description="Local file path ('sample.jpg'), relative name, s3:// URI, gs:// URI, or HTTP presigned URL",
        json_schema_extra={"example": "sample_hat.png"},
    )
    x: int = Field(
        ...,
        description="Click X coordinate on the target object",
        json_schema_extra={"example": 300},
        ge=0,
    )
    y: int = Field(
        ...,
        description="Click Y coordinate on the target object",
        json_schema_extra={"example": 200},
        ge=0,
    )
    tolerance: Optional[float] = Field(
        0.005,
        description="Polygon simplification tolerance (Ramer-Douglas-Peucker algorithm). Range: 0.001 to 0.05",
        json_schema_extra={"example": 0.005},
        ge=0.0001,
        le=0.1,
    )
    model_provider: Optional[str] = Field(
        None,
        description="Optional model engine override ('mobile_sam', 'sam2', 'opencv')",
        json_schema_extra={"example": "mobile_sam"},
    )
    level: Optional[int] = Field(
        0,
        description="Granularity level: 0 (Fine), 1 (Medium), 2 (Coarse), or None for Auto",
        json_schema_extra={"example": 0},
    )

    @field_validator("image_path")
    @classmethod
    def validate_image_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v_clean = v.strip()
        if not v_clean:
            raise ValueError("image_path cannot be empty.")
        return v_clean


class BoundaryResponse(BaseModel):
    success: bool = Field(
        True, description="Indicates if object boundary was successfully identified"
    )
    outer_boundary: List[Point] = Field(
        ...,
        description="List of (x,y) polygon vertices following the exact shape of the object",
        json_schema_extra={
            "example": [
                {"x": 180, "y": 260},
                {"x": 230, "y": 160},
                {"x": 370, "y": 160},
                {"x": 420, "y": 260},
            ]
        },
    )
    holes: List[List[Point]] = Field(
        default=[],
        description="List of inner hole polygon boundaries (e.g. cutouts inside hats, donut holes, mug handles)",
        json_schema_extra={
            "example": [
                [{"x": 300, "y": 192}, {"x": 318, "y": 210}, {"x": 282, "y": 210}]
            ]
        },
    )
    bounding_box: BoundingBox = Field(
        ..., description="Axis-aligned bounding box around the target object"
    )
    area_pixels: int = Field(
        ...,
        description="Total area of the object mask in pixels",
        json_schema_extra={"example": 24944},
    )
    confidence_score: float = Field(
        ...,
        description="Segmentation confidence score between 0.0 and 1.0",
        json_schema_extra={"example": 0.88},
    )
    image_dimensions: ImageDimensions = Field(
        ..., description="Original width and height of the image"
    )
    model_used: str = Field(
        ...,
        description="Name of the segmentation engine used",
        json_schema_extra={"example": "MobileSAM"},
    )
    processing_time_ms: float = Field(
        ...,
        description="End-to-end processing time in milliseconds",
        json_schema_extra={"example": 14.5},
    )
    error: Optional[str] = Field(
        None, description="Error message detail if success is False"
    )
