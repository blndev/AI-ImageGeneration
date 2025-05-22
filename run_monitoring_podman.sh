#!/bin/bash

# Create necessary directories
mkdir -p monitoring/grafana-data
mkdir -p monitoring/prometheus-data

# Create Prometheus config if it doesn't exist
if [ ! -f monitoring/prometheus.yml ]; then
    cat > monitoring/prometheus.yml << EOL
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'flux-generator-app'
    static_configs:
      - targets: ['host.containers.internal:9101']  # Adjust port if needed
EOL
    echo "Created Prometheus configuration file"
fi

# Function to check if a container exists
container_exists() {
    podman container exists "$1" 2>/dev/null
    return $?
}

# Function to check if a container is running
container_running() {
    [ "$(podman container inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
    return $?
}

# Handle Prometheus container
if container_exists "prometheus"; then
    echo "Prometheus container exists"
    if ! container_running "prometheus"; then
        echo "Starting existing Prometheus container"
        podman start prometheus
    else
        echo "Prometheus container is already running"
    fi
else
    echo "Creating and starting Prometheus container"
    podman run -d \
        --name prometheus \
        --network=host \
        --userns keep-id \
        -v ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:Z \
        -v ./monitoring/prometheus-data:/prometheus:Z \
        docker.io/prom/prometheus:latest
fi

# Handle Grafana container
if container_exists "grafana"; then
    echo "Grafana container exists"
    if ! container_running "grafana"; then
        echo "Starting existing Grafana container"
        podman start grafana
    else
        echo "Grafana container is already running"
    fi
else
    echo "Creating and starting Grafana container"
    podman run -d \
        --name grafana \
        --network=host \
        --userns keep-id \
        -v ./monitoring/grafana-data:/var/lib/grafana:Z \
        -e "GF_AUTH_ANONYMOUS_ENABLED=true" \
        -e "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin" \
        -e "GF_AUTH_DISABLE_LOGIN_FORM=true" \
        docker.io/grafana/grafana:latest
fi

echo -e "\nMonitoring services status:"
echo "----------------------------"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"
echo -e "\nContainers status:"
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "prometheus|grafana"
