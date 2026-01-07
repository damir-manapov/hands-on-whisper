output "vm_ip" {
  description = "Public IP address of the GPU VM"
  value       = openstack_networking_floatingip_v2.whisper.address
}

output "ssh_command" {
  description = "SSH command to connect to the VM"
  value       = "ssh root@${openstack_networking_floatingip_v2.whisper.address}"
}

output "vm_specs" {
  description = "VM specifications"
  value = {
    cpu       = var.cpu_count
    ram_gb    = var.ram_gb
    disk_gb   = var.disk_size_gb
    disk_type = var.disk_type
    region    = var.region
    zone      = var.availability_zone
  }
}

output "project_id" {
  description = "Selectel project ID"
  value       = selectel_vpc_project_v2.whisper.id
}

output "wait_for_ready" {
  description = "Command to wait for cloud-init to complete"
  value       = "ssh root@${openstack_networking_floatingip_v2.whisper.address} 'while [ ! -f /root/cloud-init-ready ]; do echo \"Waiting for setup...\"; sleep 10; done; echo \"Ready!\"'"
}

output "run_benchmark" {
  description = "Command to run Whisper optimization"
  value       = "ssh root@${openstack_networking_floatingip_v2.whisper.address} 'cd /root/hands-on-whisper && uv run python src/transcribe.py optimize calls/sherbakov_call.wav -l ru --n-trials 50'"
}
