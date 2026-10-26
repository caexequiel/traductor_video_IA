#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual..."
    uv venv --python 3.11 .venv
fi

# Ejecutar la aplicación usando el Python del entorno virtual
exec .venv/bin/python run.py
