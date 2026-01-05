#!/bin/bash
set -e

echo "=== Checking for secrets with gitleaks ==="
gitleaks detect --source . -v

echo "=== Checking for outdated dependencies ==="
# Extract direct deps from pyproject.toml
DIRECT_DEPS=$(grep '>=' pyproject.toml | grep -v requires-python | sed 's/.*"\([a-zA-Z0-9_-]*\)>=.*/\1/' | tr '\n' ' ')
OUTDATED=$(uv pip list --outdated 2>/dev/null | tail -n +3)

if [ -n "$OUTDATED" ]; then
  echo "All outdated packages:"
  echo "$OUTDATED"
  echo ""

  FAILED=""
  for dep in $DIRECT_DEPS; do
    if echo "$OUTDATED" | grep -qi "^$dep "; then
      FAILED="$FAILED $dep"
    fi
  done

  if [ -n "$FAILED" ]; then
    echo "Outdated direct dependencies found:$FAILED"
    exit 1
  fi
  echo "(Transitive dependency updates are informational only)"
else
  echo "All dependencies up to date"
fi

echo "=== Checking for vulnerabilities ==="
uv run pip-audit

echo "=== Health checks passed ==="
