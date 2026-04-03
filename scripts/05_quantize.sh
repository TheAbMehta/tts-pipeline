#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="en-us-ljspeech-medium"
OUTPUT_DIR="output"
INPUT_MODEL="${OUTPUT_DIR}/${MODEL_NAME}.onnx"

if [ ! -f "$INPUT_MODEL" ]; then
    echo "Error: Run 04_export_onnx.sh first"
    exit 1
fi

echo "=== Quantizing ONNX Model ==="

python src/quantize_model.py \
    --input "$INPUT_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --model-name "$MODEL_NAME"

echo ""
echo "Models in ${OUTPUT_DIR}/:"
ls -lh "${OUTPUT_DIR}/"*.onnx 2>/dev/null || true
