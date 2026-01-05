#!/bin/bash
set -e

./check.sh
./health.sh

echo "=== All checks passed ==="
