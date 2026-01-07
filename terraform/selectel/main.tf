terraform {
  required_version = ">= 1.0"

  required_providers {
    selectel = {
      source  = "selectel/selectel"
      version = "~> 7.0"
    }
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 3.0"
    }
  }
}

# Selectel provider for project management
provider "selectel" {
  domain_name = var.selectel_domain
  username    = var.selectel_username
  password    = var.selectel_password
  auth_region = var.region
  auth_url    = "https://cloud.api.selcloud.ru/identity/v3/"
}

# Create a project for the benchmark VM
resource "selectel_vpc_project_v2" "whisper" {
  name = "whisper-${var.environment_name}"

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [name]
  }
}

# Create a user for OpenStack access
resource "selectel_iam_serviceuser_v1" "whisper" {
  name     = "whisper-${var.environment_name}"
  password = var.selectel_openstack_password
  role {
    role_name  = "member"
    scope      = "project"
    project_id = selectel_vpc_project_v2.whisper.id
  }
}

# Add SSH key to the project
resource "selectel_vpc_keypair_v2" "whisper" {
  name       = "whisper-key"
  public_key = file(var.ssh_public_key_path)
  user_id    = selectel_iam_serviceuser_v1.whisper.id
}

# OpenStack provider for compute resources
provider "openstack" {
  auth_url    = "https://cloud.api.selcloud.ru/identity/v3"
  domain_name = var.selectel_domain
  tenant_id   = selectel_vpc_project_v2.whisper.id
  user_name   = selectel_iam_serviceuser_v1.whisper.name
  password    = var.selectel_openstack_password
  region      = var.region
}

# Get OS image (use GPU-optimized for GPU servers)
data "openstack_images_image_v2" "ubuntu" {
  name        = var.image_name
  most_recent = true

  depends_on = [
    selectel_vpc_project_v2.whisper,
    selectel_iam_serviceuser_v1.whisper
  ]
}

# Custom flavor for CPU-only mode (when use_gpu = false)
resource "openstack_compute_flavor_v2" "whisper" {
  count = var.use_gpu ? 0 : 1

  name      = "whisper-${var.cpu_count}vcpu-${var.ram_gb}gb"
  vcpus     = var.cpu_count
  ram       = var.ram_gb * 1024
  disk      = 0 # Using network boot disk
  is_public = false

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    selectel_vpc_project_v2.whisper,
    selectel_iam_serviceuser_v1.whisper
  ]
}

# For GPU mode: use fixed flavor ID from Selectel
# GPU flavor IDs are pre-defined by Selectel (e.g., GL2.X-XXXXX-X-XGPU)
# Use `openstack flavor list | grep GL` to find available GPU flavors

# Create network
resource "openstack_networking_network_v2" "whisper" {
  name = "whisper-network"

  depends_on = [
    selectel_vpc_project_v2.whisper,
    selectel_iam_serviceuser_v1.whisper
  ]
}

resource "openstack_networking_subnet_v2" "whisper" {
  name            = "whisper-subnet"
  network_id      = openstack_networking_network_v2.whisper.id
  cidr            = "10.0.0.0/24"
  dns_nameservers = ["188.93.16.19", "188.93.17.19"]
}

# Router for external connectivity
data "openstack_networking_network_v2" "external" {
  external = true

  depends_on = [
    selectel_vpc_project_v2.whisper,
    selectel_iam_serviceuser_v1.whisper
  ]
}

resource "openstack_networking_router_v2" "whisper" {
  name                = "whisper-router"
  external_network_id = data.openstack_networking_network_v2.external.id
}

resource "openstack_networking_router_interface_v2" "whisper" {
  router_id = openstack_networking_router_v2.whisper.id
  subnet_id = openstack_networking_subnet_v2.whisper.id
}

# Security group
resource "openstack_networking_secgroup_v2" "whisper" {
  name        = "whisper-secgroup"
  description = "Security group for Whisper GPU benchmark VM"

  depends_on = [
    selectel_vpc_project_v2.whisper,
    selectel_iam_serviceuser_v1.whisper
  ]
}

resource "openstack_networking_secgroup_rule_v2" "ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.whisper.id
}

# Boot volume with SSD for fast model loading
resource "openstack_blockstorage_volume_v3" "boot" {
  name              = "whisper-boot"
  size              = var.disk_size_gb
  image_id          = data.openstack_images_image_v2.ubuntu.id
  volume_type       = "${var.disk_type}.${var.availability_zone}"
  availability_zone = var.availability_zone
}

# Create a port for the VM
resource "openstack_networking_port_v2" "whisper" {
  name           = "whisper-port"
  network_id     = openstack_networking_network_v2.whisper.id
  admin_state_up = true

  fixed_ip {
    subnet_id = openstack_networking_subnet_v2.whisper.id
  }

  security_group_ids = [openstack_networking_secgroup_v2.whisper.id]

  depends_on = [
    selectel_vpc_project_v2.whisper,
    selectel_iam_serviceuser_v1.whisper
  ]
}

# Compute instance
resource "openstack_compute_instance_v2" "whisper" {
  name              = "whisper-${var.environment_name}"
  # Use GPU flavor ID if GPU mode, otherwise use custom CPU flavor
  flavor_id         = var.use_gpu ? var.gpu_flavor_id : openstack_compute_flavor_v2.whisper[0].id
  key_pair          = selectel_vpc_keypair_v2.whisper.name
  availability_zone = var.availability_zone
  user_data         = file("${path.module}/cloud-init.yaml")

  network {
    port = openstack_networking_port_v2.whisper.id
  }

  block_device {
    uuid                  = openstack_blockstorage_volume_v3.boot.id
    source_type           = "volume"
    destination_type      = "volume"
    boot_index            = 0
    delete_on_termination = true
  }

  lifecycle {
    ignore_changes = [image_id]
  }

  vendor_options {
    ignore_resize_confirmation = true
  }

  depends_on = [openstack_networking_router_interface_v2.whisper]
}

# Floating IP for external access
resource "openstack_networking_floatingip_v2" "whisper" {
  pool = "external-network"

  depends_on = [
    selectel_vpc_project_v2.whisper,
    selectel_iam_serviceuser_v1.whisper
  ]
}

resource "openstack_networking_floatingip_associate_v2" "whisper" {
  floating_ip = openstack_networking_floatingip_v2.whisper.address
  port_id     = openstack_networking_port_v2.whisper.id

  depends_on = [openstack_networking_router_interface_v2.whisper]
}
