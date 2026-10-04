# Processor record — boundary_service (Art.30(2))

- **Role:** internal processor acting only on controller (`interactive_gallery`) instructions.
- **Subject matter:** image segmentation embeddings for click-to-tag.
- **Data:** `photo_id`, S3 image URI, click `(x,y)`, ViT feature maps. No direct user accounts.
- **Controller obligations assumed:** ownership checks before every call (gallery enforces
  owner-only segment/encode/delete); DSR orchestration (export/delete fan-out) lives in gallery.
- **Assistance hooks (Art.28(3)(e)(g)):** `POST /encode`, `POST /boundary` accept optional
  `owner_id` audit passthrough; `DELETE /embedding/{photo_id}` returns 500 on durable-delete
  failure so the controller retries (never silent success).
- **Security (Art.32):** `X-Internal-Token` on all mutating routes; CORS wildcard never combined
  with credentials; S3 SSE-S3 requested (fallback for SSE-less dev endpoints); generic 500 details;
  Redis URL never logged with credentials.
- **Retention:** Redis hot 15min; S3 until controller photo/account delete. No secondary use.
- **Sub-processors:** object storage (S3/R2/MinIO) + Redis host per env. No analytics, no mailers.
- **Breach:** notify controller without undue delay; preserve logs; see gallery `BREACH_RUNBOOK.md`.
