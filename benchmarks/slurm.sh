#!/bin/bash

# Configuration
ACCOUNT="nih@h100"
CONSTRAINT="h100"
OUTPUT_DIR="results/scaling"
mkdir -p "$OUTPUT_DIR"

# Common SBATCH arguments base
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=01:00:00 --exclusive"

# Function to submit benchmarks for a backend
run_benchmarks() {
    local BACKEND=$1
    echo "Launching benchmarks for backend: $BACKEND"

    # Configurations: Nodes GPUs_per_Node Pdim_X Pdim_Y
    CONFIGS=(
        "1 1 1 1"
        "1 2 1 2"
        "1 4 2 2"
        "2 4 2 4"
        "4 4 4 4"
        "8 4 4 8"
        "16 4 8 8"
        "32 4 8 16"
        "64 4 16 16"
    )

    for CONFIG in "${CONFIGS[@]}"; do
        read -r NODES GPUS_PER_NODE PPX PPY <<< "$CONFIG"
        TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

        # Helper to submit a specific decomposition
        submit_job() {
            local PX=$1
            local PY=$2

            local JOB_NAME="bench_${BACKEND}_N${NODES}_G${GPUS_PER_NODE}_${PX}x${PY}"

            echo "Submitting $JOB_NAME (Nodes: $NODES, GPUs/Node: $GPUS_PER_NODE, Grid: ${PX}x${PY})"

            sbatch $BASE_SBATCH_ARGS \
                        --nodes=$NODES \
                        --gres=gpu:$GPUS_PER_NODE \
                        --tasks-per-node=$GPUS_PER_NODE \
                        --job-name="$JOB_NAME" \
                        $SLURM_SCRIPT python benchmarks/bench.py \
                        --pdims $PX $PY \
                        --local_shape 64 128 128 \
                        -b "$BACKEND" \
                        -n "$NODES" \
                        -o "$OUTPUT_DIR" \
                        -pr float32 \
                        -i 100 \
                        -c
        }

        # 1. Pencil (Square-ish) Decomposition from Config
        submit_job $PPX $PPY

        # 2. Slab Decompositions (if total GPUs > 1)
        if [ "$TOTAL_GPUS" -gt 1 ]; then
            # Slab Y (1 x Total)
            # Avoid duplicate if Pencil was already 1 x Total
            if [ "$PPX" != "1" ] || [ "$PPY" != "$TOTAL_GPUS" ]; then
                submit_job 1 $TOTAL_GPUS
            fi

            # Slab X (Total x 1)
            # Avoid duplicate if Pencil was already Total x 1
            if [ "$PPX" != "$TOTAL_GPUS" ] || [ "$PPY" != "1" ]; then
                submit_job $TOTAL_GPUS 1
            fi
        fi
    done
}

# Run for both backends
run_benchmarks "jax"
run_benchmarks "cudecomp"
