import cv2
import numpy as np
from pathlib import Path

def create_sample_image():
    storage_dir = Path(__file__).parent / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    img_path = storage_dir / "sample_hat.png"

    # Create a 600x600 canvas with a light background
    canvas = np.ones((600, 600, 3), dtype=np.uint8) * 240

    # Draw a person head (light peach circle)
    cv2.circle(canvas, (300, 350), 90, (180, 200, 240), -1)

    # Draw a hat on top (vibrant dark blue polygon hat with a hole/star cutout)
    hat_poly = np.array([
        [180, 260],
        [420, 260],
        [370, 160],
        [230, 160]
    ], np.int32)
    cv2.fillPoly(canvas, [hat_poly], (180, 50, 40))

    # Hat brim (ellipse)
    cv2.ellipse(canvas, (300, 260), (140, 25), 0, 0, 360, (180, 50, 40), -1)

    # Cutout/hole in hat center (white circle hole inside hat)
    cv2.circle(canvas, (300, 210), 18, (240, 240, 240), -1)

    cv2.imwrite(str(img_path), canvas)
    print(f"Sample test image created at: {img_path}")

if __name__ == "__main__":
    create_sample_image()
