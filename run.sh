#!/usr/bin/env bash
set -e

# Terminate all child processes on exit or interrupt (Ctrl+C)
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM

echo "Starting Backend (FastAPI on http://localhost:8000)..."
PYTHONPATH=. uvicorn backend.app.main:app --reload &

echo "Starting Frontend (Vite on http://localhost:5173)..."
(cd frontend && npm run dev) &

wait
