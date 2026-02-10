#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-8100}"

cd "$PROJECT_DIR"

echo "================================================"
echo "  16-Bit Portrait Generator — Local Test"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.9+."
  exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PY_VERSION"

# Install minimal deps (skip heavy GPU packages)
echo ""
echo "Installing dependencies (lightweight, no GPU)..."
python3 -m pip install -q \
  fastapi \
  'uvicorn[standard]' \
  python-multipart \
  Pillow \
  numpy \
  opencv-python-headless \
  scikit-learn \
  pydantic \
  pydantic-settings \
  mediapipe \
  2>&1 | grep -v "already satisfied" | grep -v "^$" || true

echo ""
echo "Dependencies installed."

# Find uvicorn
UVICORN="python3 -m uvicorn"
if command -v uvicorn &>/dev/null; then
  UVICORN="uvicorn"
fi

# Kill any existing process on the port
if lsof -ti :"$PORT" &>/dev/null; then
  echo "Port $PORT in use, stopping existing process..."
  kill $(lsof -ti :"$PORT") 2>/dev/null || true
  sleep 1
fi

echo ""
echo "================================================"
echo "  Starting server on http://localhost:$PORT"
echo "  Browser pipeline: works immediately (client-side)"
echo "  Server pipeline:  algorithmic fallback (no GPU)"
echo "  Press Ctrl+C to stop"
echo "================================================"
echo ""

# Open browser after a short delay
(sleep 2 && open "http://localhost:$PORT" 2>/dev/null || true) &

# Start the server
exec $UVICORN app.main:app --host 0.0.0.0 --port "$PORT" --reload
