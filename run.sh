#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <GPU_NUMBER> <CONTAINER_NAME> <FOLDER_NAME> <PYTHON_SCRIPT>"
    exit 1
fi

# Assign the provided arguments to variables
GPU_NUMBER="$1"
CONTAINER_NAME="$2"
FOLDER_NAME="$3"
PYTHON_SCRIPT="$4"

# Build the container image
podman build . -t deepmol_case_study

# Run the container with the specified Python script as the command
podman run --rm -v /home/jcorreia/deepmol_case_studies/scripts/tdc/"$FOLDER_NAME"/:/workspace/scripts/tdc/"$FOLDER_NAME"/:z -d --device nvidia.com/gpu="$GPU_NUMBER" --security-opt=label=disable --name=="$CONTAINER_NAME" deepmol_case_study /bin/bash -c "python $PYTHON_SCRIPT > output.txt"