#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Deploy F1 OmniSense backend to GCP Cloud Run
# ─────────────────────────────────────────────────────────────
# Prerequisites:
#   1. gcloud CLI installed and authenticated: gcloud auth login
#   2. A GCP project selected: gcloud config set project <PROJECT_ID>
#   3. Artifact Registry API enabled
#   4. Cloud Run API enabled
#
# Usage:
#   chmod +x deploy-backend.sh
#   ./deploy-backend.sh
# ─────────────────────────────────────────────────────────────

REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-f1-omnisense-api}"
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
  echo "ERROR: No GCP project set. Run: gcloud config set project <PROJECT_ID>"
  exit 1
fi

REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/f1-omnisense"
IMAGE="${REPO}/backend:latest"

echo "═══════════════════════════════════════════"
echo "  F1 OmniSense — Backend Deploy to Cloud Run"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Image:    ${IMAGE}"
echo "═══════════════════════════════════════════"

# ── Step 1: Create Artifact Registry repo (if not exists) ──
echo ""
echo "▸ Ensuring Artifact Registry repo exists..."
gcloud artifacts repositories describe f1-omnisense \
  --location="${REGION}" --format="value(name)" 2>/dev/null || \
gcloud artifacts repositories create f1-omnisense \
  --repository-format=docker \
  --location="${REGION}" \
  --description="F1 OmniSense Docker images"

# ── Step 2: Configure Docker auth ─────────────────────────
echo "▸ Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── Step 3: Build the image ───────────────────────────────
echo "▸ Building backend image..."
docker build -t "${IMAGE}" -f pipeline/Dockerfile .

# ── Step 4: Push to Artifact Registry ─────────────────────
echo "▸ Pushing image to Artifact Registry..."
docker push "${IMAGE}"

# ── Step 5: Deploy to Cloud Run ───────────────────────────
echo "▸ Deploying to Cloud Run..."

# Build a YAML env-vars file from .env (handles URLs and special chars)
ENV_FILE=$(mktemp /tmp/env-vars-XXXXXX.yaml)
API_PORT="8000"   # default
if [ -f .env ]; then
  while IFS='=' read -r key value; do
    # Skip comments, empty lines, malformed keys
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    [[ "$key" =~ [[:space:]] ]] && continue
    # Capture API_PORT for Cloud Run --port flag
    [[ "$key" == "API_PORT" ]] && API_PORT="$value"
    # Quote the value to handle special chars
    echo "${key}: '${value}'" >> "${ENV_FILE}"
  done < .env
fi
# Ensure API_PORT is set
grep -q "^API_PORT:" "${ENV_FILE}" || echo "API_PORT: '${API_PORT}'" >> "${ENV_FILE}"

echo "  Using port: ${API_PORT}"

gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --platform=managed \
  --port="${API_PORT}" \
  --memory=4Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=3 \
  --timeout=300s \
  --concurrency=10 \
  --startup-cpu-boost \
  --allow-unauthenticated \
  --env-vars-file="${ENV_FILE}"

rm -f "${ENV_FILE}"

# ── Step 6: Get the URL ───────────────────────────────────
BACKEND_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --format="value(status.url)")

echo ""
echo "═══════════════════════════════════════════"
echo "  Backend deployed!"
echo "  URL: ${BACKEND_URL}"
echo ""
echo "  Next: Set this as BACKEND_URL in Vercel:"
echo "    vercel env add BACKEND_URL"
echo "    → paste: ${BACKEND_URL}"
echo "═══════════════════════════════════════════"
