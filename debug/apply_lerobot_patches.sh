#!/bin/bash
# Apply lerobot patches for BEHAVIOR-1K challenge

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_VERSION=$(ls "$OPENPI_ROOT/.venv/lib/" | grep "python3\." | head -1)

LEROBOT_INSTALL_DIR="$OPENPI_ROOT/.venv/lib/$PYTHON_VERSION/site-packages/lerobot/datasets"
PATCH_SOURCE_DIR="$SCRIPT_DIR/lerobot_patches"

echo "Applying lerobot patches..."
echo "Install dir: $LEROBOT_INSTALL_DIR"

if [ ! -d "$LEROBOT_INSTALL_DIR" ]; then
    echo "Error: lerobot installation not found"
    exit 1
fi

cp "$PATCH_SOURCE_DIR/lerobot_dataset.py" "$LEROBOT_INSTALL_DIR/lerobot_dataset.py"
echo "v Patched lerobot_dataset.py"

cp "$PATCH_SOURCE_DIR/utils.py" "$LEROBOT_INSTALL_DIR/utils.py"
echo "v Patched utils.py"

echo "Done"
