# Selectel Account Credentials
# Set via environment variables:
#   export TF_VAR_selectel_domain="123456"
#   export TF_VAR_selectel_username="your-username"
#   export TF_VAR_selectel_password="your-password"
#   export TF_VAR_selectel_openstack_password="your-openstack-password"

variable "selectel_domain" {
  description = "Selectel account domain (account ID)"
  type        = string
  default     = null
}

variable "selectel_username" {
  description = "Selectel username"
  type        = string
  default     = null
}

variable "selectel_password" {
  description = "Selectel password"
  type        = string
  sensitive   = true
  default     = null
}

variable "selectel_openstack_password" {
  description = "Password for OpenStack service user"
  type        = string
  sensitive   = true
  default     = null
}

# Environment
variable "environment_name" {
  description = "Name suffix for resources (e.g., 'gpu-t4', 'gpu-a100')"
  type        = string
  default     = "gpu-benchmark"
}

variable "region" {
  description = "Selectel region (ru-7 has GPU availability)"
  type        = string
  default     = "ru-7"
}

variable "availability_zone" {
  description = "Availability zone"
  type        = string
  default     = "ru-7b"
}

# GPU Configuration
variable "gpu_flavor" {
  description = "GPU flavor name. Options: 'SL1.4-16384.1xT4', 'SL1.8-32768.1xA30', 'SL1.12-122880.1xA100'"
  type        = string
  default     = "SL1.4-16384.1xT4" # T4 with 4 vCPU, 16GB RAM - cheapest option

  # Note: Flavor names may vary. Check available flavors with:
  #   openstack flavor list --long
  # Common GPU flavors at Selectel:
  #   - SL1.4-16384.1xT4     : 4 vCPU, 16GB RAM, 1x NVIDIA T4 (16GB)
  #   - SL1.8-32768.1xA30    : 8 vCPU, 32GB RAM, 1x NVIDIA A30 (24GB)
  #   - SL1.12-122880.1xA100 : 12 vCPU, 120GB RAM, 1x NVIDIA A100 (40GB)
}

variable "disk_size_gb" {
  description = "Boot disk size in GB (need space for models ~20GB)"
  type        = number
  default     = 100
}

variable "disk_type" {
  description = "Disk type: fast, universal2, universal, basicssd, basic"
  type        = string
  default     = "fast"

  validation {
    condition     = contains(["fast", "universal2", "universal", "basicssd", "basic"], var.disk_type)
    error_message = "disk_type must be one of: fast, universal2, universal, basicssd, basic"
  }
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key file"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}
