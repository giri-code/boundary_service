from setuptools import setup, find_packages

setup(
    name="boundary_service",
    version="1.0.0",
    description="Pluggable Object Boundary Detection Microservice Module (MobileSAM, OpenCV, S3/GCS)",
    author="Antigravity",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "opencv-python-headless",
        "numpy",
        "pillow",
        "torch",
        "torchvision",
        "requests",
    ],
)
