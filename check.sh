#!/bin/bash
set -e

echo "=== Formatting ==="
uv run ruff format .

echo "=== Linting ==="
uv run ruff check --fix .

echo "=== Running tests ==="
uv run pytest

echo "=== All checks passed ==="
