#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== TTS Pipeline Setup ==="

# Build Docker image
echo "Building Docker image..."
docker build -t tts-pipeline .

echo ""
echo "Setup complete. Run the pipeline with:"
echo "  docker run --gpus all -it -v \$(pwd)/data:/workspace/tts-pipeline/data \\"
echo "    -v \$(pwd)/training:/workspace/tts-pipeline/training \\"
echo "    -v \$(pwd)/checkpoints:/workspace/tts-pipeline/checkpoints \\"
echo "    -v \$(pwd)/output:/workspace/tts-pipeline/output \\"
echo "    tts-pipeline bash"
echo ""
echo "Then inside the container, run scripts in order:"
echo "  bash scripts/01_download_data.sh"
echo "  bash scripts/02_preprocess.sh"
echo "  bash scripts/03_train.sh"
echo "  bash scripts/04_export_onnx.sh"
echo "  bash scripts/05_quantize.sh"
echo "  bash scripts/06_test_inference.sh"
