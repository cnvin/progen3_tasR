#!/bin/bash

set -euo pipefail

model_name=$1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR=$SCRIPT_DIR/outputs/$model_name

mkdir -p "$OUTPUT_DIR"

torchrun --nproc-per-node=gpu -m progen3.tools.generate \
    --prompt-file "$SCRIPT_DIR/prompts.csv" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$model_name" \
    --n-per-prompt 5000 \
    --fsdp \
    --temperature 0.85 \
    --top-p 0.95
