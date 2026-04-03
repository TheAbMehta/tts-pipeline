#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT_DIR="checkpoints"
TRAINING_DIR="training"
OUTPUT_DIR="output"
MODEL_NAME="en-us-ljspeech-medium"

mkdir -p "$OUTPUT_DIR"

CKPT=$(find "$CHECKPOINT_DIR" -name "*.ckpt" -type f | sort -V | tail -1)

if [ -z "$CKPT" ]; then
    echo "Error: No checkpoint found in ${CHECKPOINT_DIR}/"
    echo "Run 03_train.sh first"
    exit 1
fi

echo "=== Exporting ONNX ==="
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUTPUT_DIR}/${MODEL_NAME}.onnx"

python -m piper_train.export_onnx \
    "$CKPT" \
    "${OUTPUT_DIR}/${MODEL_NAME}.onnx"

if [ -f "${OUTPUT_DIR}/${MODEL_NAME}.onnx" ]; then
    SIZE=$(du -h "${OUTPUT_DIR}/${MODEL_NAME}.onnx" | cut -f1)
    echo ""
    echo "Export complete."
    echo "  Model: ${OUTPUT_DIR}/${MODEL_NAME}.onnx (${SIZE})"
    echo "  Config: ${OUTPUT_DIR}/${MODEL_NAME}.onnx.json"
else
    echo "Error: ONNX export failed"
    exit 1
fi
