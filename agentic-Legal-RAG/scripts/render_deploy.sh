#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${RENDER_DEPLOY_HOOK_URL:-}" ]]; then
  echo "Triggering Render deploy hook..."
  curl -fsS -X POST "$RENDER_DEPLOY_HOOK_URL" >/dev/null
  echo "Render deployment triggered with deploy hook."
  exit 0
fi

if [[ -n "${RENDER_API_KEY:-}" && -n "${RENDER_SERVICE_ID:-}" ]]; then
  echo "Triggering Render deployment via Render API..."
  curl -fsS \
    -X POST \
    -H "Authorization: Bearer ${RENDER_API_KEY}" \
    -H "Content-Type: application/json" \
    "https://api.render.com/v1/services/${RENDER_SERVICE_ID}/deploys" \
    -d '{}' >/dev/null
  echo "Render deployment triggered with API key and service ID."
  exit 0
fi

echo "Missing Render credentials."
echo "Set either:"
echo "  - RENDER_DEPLOY_HOOK_URL"
echo "or both:"
echo "  - RENDER_API_KEY"
echo "  - RENDER_SERVICE_ID"
exit 1
