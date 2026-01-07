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
  description = "Availability zone (ru-7a for GPU, ru-7b for CPU)"
  type        = string
  default     = "ru-7a" # GPU available in ru-7a
}

# VM Configuration
variable "use_gpu" {
  description = "Whether to use GPU flavor (true) or custom CPU flavor (false)"
  type        = bool
  default     = true
}

variable "gpu_flavor_id" {
  description = <<-EOT
    Fixed GPU flavor ID from Selectel. Common options for ru-7:
    - Tesla T4 1 GPU:  4 vCPU, 32GB RAM  -> check with `openstack flavor list | grep GL`
    - Tesla T4 2 GPU: 12 vCPU, 175GB RAM -> 3042
    Use `openstack flavor list` to see available GPU flavors in your pool.
  EOT
  type        = string
  default     = null # Will be required when use_gpu = true
}

variable "cpu_count" {
  description = "Number of vCPUs (only used when use_gpu = false)"
  type        = number
  default     = 8
}

variable "ram_gb" {
  description = "RAM in GB (only used when use_gpu = false)"
  type        = number
  default     = 32
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

variable "image_name" {
  description = <<-EOT
    OS image name. For GPU servers, use GPU-optimized images with pre-installed NVIDIA drivers:
    - "Ubuntu 22.04 LTS 64-bit GPU Driver 580" (recommended)
    - "Ubuntu 22.04 LTS 64-bit GPU Driver 580 Docker" (with Docker)
    - "Ubuntu 22.04 LTS Machine Learning 64-bit" (ML tools included)
    - "Ubuntu 24.04 LTS 64-bit" (requires manual driver installation)
  EOT
  type        = string
  default     = "Ubuntu 22.04 LTS 64-bit GPU Driver 580"
}
