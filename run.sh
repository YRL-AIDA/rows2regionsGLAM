#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON_BIN:-python3}"
exec "$PYTHON" scripts/run.py "$@"
