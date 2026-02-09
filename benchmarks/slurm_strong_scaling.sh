#!/bin/bash

# Configuration
ACCOUNT="XXX"
CONSTRAINT="h100"
OUTPUT_DIR="results/strong_scaling"
mkdir -p "$OUTPUT_DIR"

# Check for SLURM_SCRIPT environment variable
if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    echo "Please set it to the path of your sbatch launcher script."
    exit 1
fi

# Common SBATCH arguments base
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=01:00:00 --exclusive"

# Function to submit benchmarks for a backend
run_benchmarks() {
    local BACKEND=$1
    local PRECISION=$2
    echo "Launching benchmarks for backend: $BACKEND"

    # Configurations: Nodes GPUs_per_Node Pdim_X Pdim_Y
    CONFIGS=(
        "1 2 1 2"
        # Four GPUS
        "1 4 2 2"
        "1 4 1 4"
        "1 4 4 1"
        # Eight GPUS
        "2 4 2 4"
        "2 4 4 2"
        "2 4 1 8"
        "2 4 8 1"
        # Sixteen GPUS
        "4 4 4 4"
        "4 4 2 8"
        "4 4 8 2"
        "4 4 1 16"
        "4 4 16 1"
        # Thirty-Two GPUS
        "8 4 4 8"
        "8 4 8 4"
        "8 4 1 32"
        "8 4 32 1"
        # Sixty-Four GPUS
        "16 4 8 8"
        "16 4 4 16"
        "16 4 16 4"
        "16 4 1 64"
        "16 4 64 1"
        # One Hundred Twenty-Eight GPUS
        "32 4 8 16"
        "32 4 16 8"
        "32 4 1 128"
        "32 4 128 1"
        # 256 GPUs
        "64 4 16 16"
        "64 4 8 32"
        "64 4 1 256"
        "64 4 256 1"
    )

    # Local Sizes to iterate
    # Range: 64^3 to 512^3
    SHAPES=(
        "128 128 128"
        "256 256 256"
        "512 512 512"
        "1024 1024 1024"
        "2048 2048 2048"
        "4096 4096 4096"
    )

    for CONFIG in "${CONFIGS[@]}"; do
        read -r NODES GPUS_PER_NODE PX PY <<< "$CONFIG"

        for SHAPE in "${SHAPES[@]}"; do
            # Convert spaces to 'x' for job name, e.g., 64x64x64
            SHAPE_NAME=${SHAPE// /x}

            local JOB_NAME="strong_bench_${BACKEND}_N${NODES}_G${GPUS_PER_NODE}_${PX}x${PY}_${SHAPE_NAME}"

            echo "Submitting $JOB_NAME (Nodes: $NODES, GPUs/Node: $GPUS_PER_NODE, Grid: ${PX}x${PY}, Local: $SHAPE)"

            sbatch $BASE_SBATCH_ARGS \
                        --nodes=$NODES \
                        --gres=gpu:$GPUS_PER_NODE \
                        --tasks-per-node=$GPUS_PER_NODE \
                        --job-name="$JOB_NAME" \
                        $SLURM_SCRIPT STRONG_TRACES python bench.py \
                        --pdims $PX $PY \
                        --global_shape $SHAPE \
                        -b "$BACKEND" \
                        -n "$NODES" \
                        -o "$OUTPUT_DIR" \
                        -pr "$PRECISION" \
                        -i 5 \
                        -c
        done
    done
}

# Run for both backends
run_benchmarks "jax" "float32"
run_benchmarks "cudecomp" "float32"
run_benchmarks "jax" "float64"
run_benchmarks "cudecomp" "float64"
