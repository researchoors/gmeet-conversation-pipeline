variable "node_name" {
  type        = string
  description = "Nomad node.unique.name to pin the gmeet-pipeline allocation to."
}

variable "http_port" {
  type        = number
  description = "Static host port exposed by the gmeet-pipeline HTTP server."
}

variable "runner_command" {
  type        = string
  description = "Host-owned wrapper that starts gmeet-pipeline with local env/secrets."
}

job "gmeet-pipeline" {
  datacenters = ["researchoors"]
  type        = "service"

  group "web" {
    count = 1

    constraint {
      attribute = "${node.unique.name}"
      value     = var.node_name
    }

    network {
      port "http" {
        static = var.http_port
        to     = var.http_port
      }
    }

    task "server" {
      driver = "raw_exec"

      config {
        command = var.runner_command
      }

      env {
        GMEET_PORT = "${var.http_port}"
      }

      resources {
        cpu    = 1000
        memory = 8192
      }

      restart {
        attempts = 10
        interval = "5m"
        delay    = "15s"
        mode     = "delay"
      }

      service {
        name = "gmeet-pipeline"
        port = "http"

        check {
          name     = "http-health"
          type     = "http"
          path     = "/health"
          interval = "10s"
          timeout  = "3s"
        }
      }
    }
  }
}
