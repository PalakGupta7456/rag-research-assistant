#!/bin/sh
set -e
if [ ! -f data/papers.parquet ]; then
  echo "No corpus found: building one from arXiv (takes a few minutes)..."
  python -m rag.build
fi
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
