#!/bin/bash

# Get the script directory to make paths more reliable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# From AlphaFold3 directory, go up to the mindscience directory
MINDSCIENCE_PATH="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Check if the base directory exists
if [ ! -d "$MINDSCIENCE_PATH" ]; then
    echo "Error: MindScience path not found: $MINDSCIENCE_PATH"
    echo "Please run this script from the correct directory"
    exit 1
fi

# Function to add to PYTHONPATH if directory exists
add_to_pythonpath() {
    local dir_path="$1"
    if [ -d "$dir_path" ]; then
        export PYTHONPATH="$PYTHONPATH:$dir_path"
        echo "Added to PYTHONPATH: $dir_path"
    else
        echo "Warning: Directory not found, skipping: $dir_path"
    fi
}

add_to_pythonpath "$MINDSCIENCE_PATH"
add_to_pythonpath "$MINDSCIENCE_PATH/MindSPONGE/applications/AlphaFold3"

# Add directories to PATH
export PATH=$PATH:/hmmer/bin

# Display current PYTHONPATH
echo "Current PYTHONPATH:"
echo "$PYTHONPATH" | tr ':' '\n' | sed 's/^/  /'

echo "Environment setup completed."
