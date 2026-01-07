#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Terraform Format Check ==="
terraform fmt -check -recursive .

echo "=== Terraform Validate ==="
cd selectel
terraform init -backend=false > /dev/null 2>&1 || true
terraform validate

echo "=== All checks passed ==="
