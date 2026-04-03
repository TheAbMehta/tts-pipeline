#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAME="en-us-ljspeech-medium"
OUTPUT_DIR="output"
TEST_TEXT="${1:-The quick brown fox jumps over the lazy dog.}"

echo "=== Testing Inference ==="
echo "Text: ${TEST_TEXT}"
echo ""

echo "--- Test 1: ONNX Runtime ---"
python src/test_inference.py \
    --model "${OUTPUT_DIR}/${MODEL_NAME}.onnx" \
    --config "${OUTPUT_DIR}/${MODEL_NAME}.onnx.json" \
    --text "$TEST_TEXT" \
    --output "${OUTPUT_DIR}/test_onnxrt.wav"

if [ -f "${OUTPUT_DIR}/${MODEL_NAME}-int8.onnx" ]; then
    echo ""
    echo "--- Test 2: ONNX Runtime (INT8) ---"
    python src/test_inference.py \
        --model "${OUTPUT_DIR}/${MODEL_NAME}-int8.onnx" \
        --config "${OUTPUT_DIR}/${MODEL_NAME}.onnx.json" \
        --text "$TEST_TEXT" \
        --output "${OUTPUT_DIR}/test_onnxrt_int8.wav"
fi

echo ""
echo "--- Test 3: Sherpa-ONNX ---"
python src/test_sherpa.py \
    --model "${OUTPUT_DIR}/${MODEL_NAME}.onnx" \
    --config "${OUTPUT_DIR}/${MODEL_NAME}.onnx.json" \
    --text "$TEST_TEXT" \
    --output "${OUTPUT_DIR}/test_sherpa.wav"

echo ""
echo "=== All tests complete ==="
echo "Output files:"
ls -lh "${OUTPUT_DIR}"/test_*.wav 2>/dev/null || echo "  (no output files)"
