#!/usr/bin/env bash

# Mirror tutorials from docs to package examples directory
# This allows tutorials to be distributed with the package

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define source and destination directories
SRC_DIR="$SCRIPT_DIR/docs/tutorials"
DEST_DIR="$SCRIPT_DIR/src/statecounter/examples/tutorials"

# Remove current tutorials in distribution directory (if exists)
if [ -d "$DEST_DIR" ]; then
    rm -r "$DEST_DIR"
fi

# Create destination directory structure
mkdir -p "$DEST_DIR"

# Mirror tutorials
cp -r "$SRC_DIR"/* "$DEST_DIR"/

echo "Tutorials mirrored from $SRC_DIR to $DEST_DIR"
