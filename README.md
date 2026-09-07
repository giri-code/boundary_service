# Object Boundary Detection Microservice (`boundary_service`)

A production-ready, pluggable Python FastAPI microservice that receives image locations (local paths, AWS `s3://`, GCP `gs://`, or `http://` URLs) and click point coordinates `(x, y)`, and returns precise object polygon boundary coordinates `[{x, y}]` and inner hole geometries.

---

## 💻 Running Locally

### Option 1: Direct Python Execution (Recommended for Local Dev)

1. **Navigate to the module directory**:
   ```bash
   cd /Users/lgiri/marketing/boundary_service
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access Swagger UI in your browser**:
   Open [http://localhost:8000/docs](http://localhost:8000/docs) to test endpoints interactively.

---

### Option 2: Local Docker Container

```bash
cd /Users/lgiri/marketing/boundary_service
docker-compose up --build
```

---

## 🧪 Testing Local Images

### 1. Run Automated Test Suite
```bash
python3 -m pytest tests/
```

### 2. Run Visual Boundary Renderer on a Local Photo
Generate annotated visual output (`storage/output_boundary_visual.png`):
```bash
python3 tests/visual_test_standalone.py
```

### 3. Example Local API Call (cURL)
```bash
curl -X POST "http://localhost:8000/api/v1/boundary" \
     -H "Content-Type: application/json" \
     -d '{
           "image_path": "sample_hat.png",
           "x": 300,
           "y": 200,
           "tolerance": 0.005
         }'
```

---

## ⚙️ Local & Cloud Environment Settings (`.env`)

Default settings resolve to local disk storage (`./storage`). You can configure `.env`:

```ini
ENVIRONMENT=local
STORAGE_BACKEND=auto             # "auto", "local", "s3", "gcs"
LOCAL_STORAGE_ROOT=./storage     # Local photo storage directory
SEGMENTATION_MODEL_PROVIDER=mobile_sam # "mobile_sam", "sam2", "opencv"
```

---

## ☁️ Zero-Code-Change Cloud Deployment

When moving to **AWS** or **GCP**:
- **AWS S3**: Pass `s3://my-bucket/image.jpg` or set `STORAGE_BACKEND=s3`.
- **GCP GCS**: Pass `gs://my-bucket/image.jpg` or set `STORAGE_BACKEND=gcs`.
- **Deploy to Cloud Run**: `./deploy/gcp_cloud_run.sh`
