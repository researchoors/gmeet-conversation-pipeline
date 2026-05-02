job "gmeet-pipeline" {
  datacenters = ["researchoors"]
  type        = "service"

  group "web" {
    count = 1

    constraint {
      attribute = "${node.unique.name}"
      value     = "inference2s-Mac-Studio.local"
    }

    network {
      port "http" {
        static = 9120
        to     = 9120
      }
    }

    task "server" {
      driver = "raw_exec"

      config {
        command = "/Users/inference2/.hermes/deployments/bin/run-gmeet-pipeline.sh"
      }

      env {
        GMEET_PORT = "9120"
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
