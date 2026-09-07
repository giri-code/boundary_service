#!/usr/bin/env bash
set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-gcp-project"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="boundary-detection-service"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=== Building Docker Image for GCP Cloud Run ==="
gcloud builds submit --tag ${IMAGE_TAG} .

echo "=== Deploying to GCP Cloud Run ==="
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_TAG} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars STORAGE_BACKEND=gcs,SEGMENTATION_MODEL_PROVIDER=mobile_sam

echo "=== Deployment Successful! ==="
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'
