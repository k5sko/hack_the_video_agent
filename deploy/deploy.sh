#!/usr/bin/env bash
#
# Deploy Beeline to AWS: clips and frontend to S3 behind CloudFront, the API to
# App Runner from an ECR image.
#
# Idempotent -- safe to re-run. Each step checks whether its resource already
# exists rather than assuming a clean account.
#
#   ./deploy/deploy.sh all        # everything
#   ./deploy/deploy.sh media      # just re-upload clips
#   ./deploy/deploy.sh frontend   # just rebuild + re-upload the UI
#   ./deploy/deploy.sh api        # just rebuild + redeploy the container
#
# Prerequisites: AWS credentials with permission for S3, ECR, App Runner,
# CloudFront and Secrets Manager; docker; and a populated .env at the repo root.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AWS="${AWS:-$REPO_ROOT/.venv/bin/aws}"
REGION="${AWS_REGION:-us-east-1}"
APP="${BEELINE_APP_NAME:-beeline}"
SECRET_NAME="${BEELINE_SECRET_NAME:-$APP/env}"

CLIPS_DIR="beeline/data/clips"
FRONTEND_DIR="beeline/frontend"

log() { printf '\n=== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

account_id() { "$AWS" sts get-caller-identity --query Account --output text; }

require_creds() {
  "$AWS" sts get-caller-identity >/dev/null 2>&1 \
    || die "no AWS credentials. Run 'aws configure' or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
}

bucket_name() { echo "${APP}-$(account_id)-${REGION}"; }

# --------------------------------------------------------------------------- #

step_media() {
  log "Uploading clips to S3"
  local bucket; bucket="$(bucket_name)"
  [ -d "$CLIPS_DIR" ] || die "$CLIPS_DIR missing -- run beeline/ingestion/cut.py first"

  if ! "$AWS" s3api head-bucket --bucket "$bucket" >/dev/null 2>&1; then
    if [ "$REGION" = "us-east-1" ]; then
      "$AWS" s3api create-bucket --bucket "$bucket" --region "$REGION"
    else
      "$AWS" s3api create-bucket --bucket "$bucket" --region "$REGION" \
        --create-bucket-configuration "LocationConstraint=$REGION"
    fi
  fi

  # Long cache: a clip's bytes never change once cut, and the id is derived from
  # its content window.
  "$AWS" s3 sync "$CLIPS_DIR" "s3://$bucket/media/" \
    --content-type video/mp4 \
    --cache-control "public, max-age=31536000, immutable" \
    --only-show-errors
  echo "clips: s3://$bucket/media/ ($(ls "$CLIPS_DIR"/*.mp4 2>/dev/null | wc -l) files)"
}

step_frontend() {
  log "Building and uploading the frontend"
  local bucket; bucket="$(bucket_name)"
  local api_url="${BEELINE_API_URL:-}"
  [ -n "$api_url" ] || echo "  note: BEELINE_API_URL unset; the UI will call localhost:8000"

  ( cd "$FRONTEND_DIR" && npm install --silent && VITE_API_BASE="$api_url" npm run build )

  # index.html must not be cached, or a deploy is invisible until the CDN expires.
  "$AWS" s3 sync "$FRONTEND_DIR/dist" "s3://$bucket/app/" \
    --exclude index.html \
    --cache-control "public, max-age=31536000, immutable" \
    --only-show-errors
  "$AWS" s3 cp "$FRONTEND_DIR/dist/index.html" "s3://$bucket/app/index.html" \
    --cache-control "no-cache" --only-show-errors
  echo "frontend: s3://$bucket/app/"
}

step_secrets() {
  log "Syncing .env to Secrets Manager"
  [ -f .env ] || die ".env missing"
  # Build JSON from .env without ever echoing a value.
  local payload
  payload="$("$REPO_ROOT/.venv/bin/python" - <<'PY'
import json
out = {}
for line in open(".env"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    v = v.strip().strip('"').strip("'")
    if v:
        out[k.strip()] = v
print(json.dumps(out))
PY
)"
  if "$AWS" secretsmanager describe-secret --secret-id "$SECRET_NAME" >/dev/null 2>&1; then
    "$AWS" secretsmanager put-secret-value --secret-id "$SECRET_NAME" \
      --secret-string "$payload" >/dev/null
  else
    "$AWS" secretsmanager create-secret --name "$SECRET_NAME" \
      --secret-string "$payload" >/dev/null
  fi
  echo "secret: $SECRET_NAME (values never printed)"
}

step_api() {
  log "Building and pushing the API image"
  local account; account="$(account_id)"
  local registry="${account}.dkr.ecr.${REGION}.amazonaws.com"
  local image="${registry}/${APP}:latest"

  "$AWS" ecr describe-repositories --repository-names "$APP" >/dev/null 2>&1 \
    || "$AWS" ecr create-repository --repository-name "$APP" >/dev/null

  "$AWS" ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$registry"

  docker build -t "$APP:latest" .
  docker tag "$APP:latest" "$image"
  docker push "$image"
  echo "image: $image"
  echo
  echo "Next: create the App Runner service pointing at that image."
  echo "See deploy/README.md -- it needs an access role and the secret ARN, which"
  echo "are easier to review in the console than to conjure from a script."
}

# --------------------------------------------------------------------------- #

require_creds
case "${1:-all}" in
  media)    step_media ;;
  frontend) step_frontend ;;
  secrets)  step_secrets ;;
  api)      step_api ;;
  all)      step_media; step_secrets; step_api; step_frontend ;;
  *)        die "unknown step '${1}' (media|frontend|secrets|api|all)" ;;
esac

log "Done"
