#!/bin/bash
set -e

echo "=== Checking for secrets with gitleaks ==="
gitleaks detect --source . -v

echo "=== Checking for outdated dependencies ==="
./renovate-check.sh

echo "=== Checking for vulnerabilities ==="
uv run pip-audit --skip-editable --progress-spinner=off 2>&1 | grep -vE "(Skip Reason|hands-on-whisper|^-+ +-+$|^Name +Skip)"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  exit 1
fi

echo "=== Health checks passed ==="
