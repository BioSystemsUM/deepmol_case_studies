#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <GPU_NUMBER> <FOLDER_NAME> <PYTHON_SCRIPT>"
    exit 1
fi

# Assign the provided arguments to variables
GPU_NUMBER="$1"
FOLDER_NAME="$2"
PYTHON_SCRIPT="$3"

# Build the container image
podman build . -t deepmol_case_study

# Run the container with the specified Python script as the command
podman run -v /home/jcapela/deepmol_case_studies/scripts/benchmark_automl/data_for_benchmark/"$FOLDER_NAME"/:/workspace/scripts/benchmark_automl/data_for_benchmark/"$FOLDER_NAME"/:z -d --device nvidia.com/gpu="$GPU_NUMBER" --security-opt=label=disable --name="$FOLDER_NAME" deepmol_case_study /bin/bash -c "cd /workspace/scripts/benchmark_automl/data_for_benchmark/$FOLDER_NAME/ && python $PYTHON_SCRIPT > output.txt"
