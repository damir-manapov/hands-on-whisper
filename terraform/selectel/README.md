# Terraform for Whisper GPU Benchmarks

Provision a GPU VM on Selectel for running Whisper transcription benchmarks.

## Prerequisites

1. [Terraform](https://terraform.io/downloads) >= 1.0
2. Selectel account with GPU quota (contact support if needed)
3. SSH key pair

## GPU Options

| Flavor | GPU | VRAM | vCPU | RAM | ~Price/hr |
|--------|-----|------|------|-----|-----------|
| SL1.4-16384.1xT4 | T4 | 16GB | 4 | 16GB | ~₽50 |
| SL1.8-32768.1xA30 | A30 | 24GB | 8 | 32GB | ~₽100 |
| SL1.12-122880.1xA100 | A100 | 40GB | 12 | 120GB | ~₽200 |

Note: Prices are approximate. Check current pricing at https://selectel.ru/services/gpu/

## Setup

```bash
cd terraform/selectel

# Copy example config
cp terraform.tfvars.example terraform.tfvars

# Set credentials via environment variables
export TF_VAR_selectel_domain="your-account-id"
export TF_VAR_selectel_username="your-username"
export TF_VAR_selectel_password="your-password"
export TF_VAR_selectel_openstack_password="your-openstack-password"
```

Get credentials from https://my.selectel.ru/profile/apikeys

## Usage

```bash
# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Create VM
terraform apply

# Get SSH command
terraform output ssh_command

# Wait for cloud-init to complete
eval $(terraform output -raw wait_for_ready)

# Run benchmark
eval $(terraform output -raw run_benchmark)

# Destroy when done (important - GPU VMs are expensive!)
terraform destroy
```

## What's Installed

The cloud-init script automatically installs:
- NVIDIA drivers + CUDA 12.4
- Python 3.11+ with uv
- FFmpeg
- hands-on-whisper repo with all dependencies

**Models are downloaded on-demand** when first used (saves ~15GB download on each server creation).

## Running Benchmarks

After SSH'ing into the VM:

```bash
cd /root/hands-on-whisper

# Quick test (will download the model on first run)
uv run python src/transcribe.py transcribe calls/sherbakov_call.wav -l ru --runtime cuda

# Full optimization (50 trials)
uv run python src/transcribe.py optimize calls/sherbakov_call.wav -l ru --n-trials 50

# Copy results back to local machine
scp root@<vm-ip>:hands-on-whisper/calls/*.json ./calls/
scp root@<vm-ip>:hands-on-whisper/calls/*.md ./calls/
```

## Cost Optimization

- Use `terraform destroy` immediately after benchmarks
- T4 is cheapest and sufficient for most testing
- Consider Selectel's preemptible VMs for even lower costs

## Troubleshooting

Check NVIDIA driver status:
```bash
nvidia-smi
```

Check cloud-init logs:
```bash
cat /var/log/cloud-init-output.log
```

If NVIDIA driver isn't ready, reboot:
```bash
reboot
```

## OpenStack CLI

For finding GPU flavors and debugging:

```bash
# Set environment variables
export OS_AUTH_URL="https://cloud.api.selcloud.ru/identity/v3"
export OS_IDENTITY_API_VERSION=3
export OS_PROJECT_DOMAIN_NAME="$TF_VAR_selectel_domain"
export OS_USER_DOMAIN_NAME="$TF_VAR_selectel_domain"
export OS_PROJECT_ID="$(terraform output -raw project_id)"
export OS_USERNAME="$TF_VAR_selectel_username"
export OS_PASSWORD="$TF_VAR_selectel_password"
export OS_REGION_NAME="ru-7"

# List all GPU flavors (GL prefix)
openstack flavor list | grep GL

# Get flavor ID by name
openstack flavor show "GL2.6-24576-0-1GPU" -f value -c id

# List GPU-optimized images
openstack image list | grep -i gpu
```

Common GPU flavor naming: `GLX.vCPU-RAM-DISK-NxGPU` (e.g., `GL2.6-24576-0-1GPU` = 6 vCPU, 24GB RAM, 1 GPU)
