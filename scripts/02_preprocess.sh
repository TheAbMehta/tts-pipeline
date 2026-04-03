#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LJSPEECH_DIR="data/LJSpeech-1.1"
TRAINING_DIR="training"

if [ ! -f "${LJSPEECH_DIR}/metadata_piper.csv" ]; then
    echo "Error: Run 01_download_data.sh first"
    exit 1
fi

mkdir -p "$TRAINING_DIR"

echo "=== Preprocessing LJSpeech for VITS training ==="
echo "Language: en-us | Quality: medium | Single speaker"

python3 -m piper_train.preprocess \
    --language en-us \
    --input-dir "$LJSPEECH_DIR" \
    --output-dir "$TRAINING_DIR" \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050

if [ -f "${TRAINING_DIR}/dataset.jsonl" ]; then
    ENTRIES=$(wc -l < "${TRAINING_DIR}/dataset.jsonl")
    echo ""
    echo "Preprocessing complete."
    echo "  dataset.jsonl: ${ENTRIES} entries (expected ~13100)"
    echo "  Output dir: ${TRAINING_DIR}/"
else
    echo "Error: dataset.jsonl not found after preprocessing"
    exit 1
fi
