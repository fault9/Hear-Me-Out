#!/bin/bash
# Build the selected Vite frontend (npm workspaces monorepo).
#   APP_MODE=hmo   -> builds frontend/         (default)
#   APP_MODE=study -> builds study-frontend/
# Both import the shared library at packages/shared (consumed as source).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_MODE="${APP_MODE:-hmo}"

if [ "$APP_MODE" = "study" ]; then
  WS="study-frontend"
else
  WS="frontend"
fi

cd "$PROJECT_DIR"
echo "=== Building $WS (APP_MODE=$APP_MODE) ==="

# Install at the workspace root when node_modules is missing OR a lockfile/manifest
# changed (catches newly-added deps on a git pull).
if [ ! -d node_modules ] || [ package-lock.json -nt node_modules ] || [ package.json -nt node_modules ]; then
  echo "Installing workspace dependencies..."
  npm install
fi

npm run build -w "$WS"

echo "=== Build complete → $PROJECT_DIR/$WS/dist ==="
