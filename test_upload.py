import requests
import cv2
import numpy as np

# Create a dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.imwrite("dummy.jpg", img)

url = "http://127.0.0.1:8000/api/v1/boundary/upload"
headers = {"X-Internal-Token": "super-secret-boundary-token-123"}
with open("dummy.jpg", "rb") as f:
    files = {"file": ("dummy.jpg", f, "image/jpeg")}
    data = {"x": 50, "y": 50, "level": 0}
    res = requests.post(url, headers=headers, files=files, data=data)

print(res.status_code, res.text)
