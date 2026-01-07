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

# VM Configuration
variable "cpu_count" {
  description = "Number of vCPUs"
  type        = number
  default     = 8
}

variable "ram_gb" {
  description = "RAM in GB"
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
