#!/bin/bash
# Wrapper script to run ALOHA sim with GPU support
# Docker Compose v2.39.2 has a bug where GPU passthrough doesn't work with 'up' command
# This script uses docker run directly which works correctly

set -e

cd "$(dirname "$0")"

# Build images if needed
echo "Building Docker images..."
docker build -t openpi_server -f scripts/docker/serve_policy.Dockerfile .
docker build -t aloha_sim -f examples/aloha_sim/Dockerfile .

# Start openpi_server in background
# Note: We override the CMD to use PYTHONPATH instead of 'uv run' to avoid
# package reinstallation that breaks JAX CUDA support
echo "Starting openpi_server..."
docker run -d \
  --name aloha_openpi_server \
  --gpus device=0 \
  --network host \
  -e SERVER_ARGS="--env ALOHA_SIM" \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e IS_DOCKER=true \
  -v "$(pwd):/app" \
  -v "${HOME}/.cache/openpi:/openpi_assets" \
  openpi_server \
  /bin/bash -c "cd /app && PYTHONPATH=/app/src:/.venv/lib/python3.11/site-packages /.venv/bin/python scripts/serve_policy.py --env ALOHA_SIM"

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Start runtime container
echo "Starting runtime container..."
docker run -it --rm \
  --name aloha_runtime \
  --network host \
  --privileged \
  -v "$(pwd):/app" \
  -v "$(pwd)/data:/data" \
  aloha_sim

# Cleanup: stop server when runtime exits
echo "Stopping openpi_server..."
docker stop aloha_openpi_server
docker rm aloha_openpi_server
