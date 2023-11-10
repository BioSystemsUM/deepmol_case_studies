#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <GPU_NUMBER> <CONTAINER_NAME> <PYTHON_SCRIPT>"
    exit 1
fi

# Assign the provided arguments to variables
GPU_NUMBER="$1"
CONTAINER_NAME="$2"
PYTHON_SCRIPT="$3"

# Build the container image
podman build . -t deepmol_case_study

# Run the container with the specified Python script as the command
podman run --rm -v /home/jcorreia/deepmol_case_studies/scripts/:/workspace/scripts/:z -d --device nvidia.com/gpu="$GPU_NUMBER" --security-opt=label=disable --name=="$CONTAINER_NAME" deepmol_case_study python "$PYTHON_SCRIPT"