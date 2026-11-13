#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export LABEL_STUDIO_BASE_DATA_DIR="$PWD/data/label-studio"
export LABEL_STUDIO_COLLECT_ANALYTICS=false
export LABEL_STUDIO_LATEST_VERSION_CHECK=false
exec uv run --project tools/label-studio --locked label-studio start \
  --internal-host 127.0.0.1 --port "${LABEL_STUDIO_PORT:-8080}" --no-browser "$@"
