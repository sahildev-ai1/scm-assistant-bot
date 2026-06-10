#!/bin/bash
# start.sh — Launch FastAPI + Streamlit on a single Render web service
# Render assigns PORT env var; Streamlit runs on PORT, FastAPI on 8000 (internal)

set -e

PORT="${PORT:-10000}"

echo "=== SCM Assistant starting ==="
echo "FastAPI  → :8000 (internal)"
echo "Streamlit→ :$PORT (public)"

# Start FastAPI in background
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
echo "Waiting for FastAPI..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "FastAPI ready ✓"
    break
  fi
  sleep 2
done

# Start Streamlit (foreground, Render watches this process)
streamlit run frontend/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false

# If streamlit exits, kill fastapi too
kill $FASTAPI_PID
