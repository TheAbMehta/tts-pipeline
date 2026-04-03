#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAINING_DIR="training"
CHECKPOINT_DIR="checkpoints"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-10000}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-500}"
QUALITY="${QUALITY:-medium}"

if [ ! -f "${TRAINING_DIR}/dataset.jsonl" ]; then
    echo "Error: Run 02_preprocess.sh first"
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

echo "=== Starting VITS Training ==="
echo "Quality: ${QUALITY} (192 hidden channels)"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size: ${BATCH_SIZE} per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Checkpoint every: ${CHECKPOINT_EPOCHS} epochs"
echo ""
echo "Monitor with: tensorboard --logdir ${CHECKPOINT_DIR}/lightning_logs"
echo ""

python3 -m piper_train \
    --dataset-dir "$TRAINING_DIR" \
    --accelerator gpu \
    --devices "$NUM_GPUS" \
    --batch-size "$BATCH_SIZE" \
    --validation-split 0.01 \
    --max_epochs "$MAX_EPOCHS" \
    --checkpoint-epochs "$CHECKPOINT_EPOCHS" \
    --quality "$QUALITY" \
    --default_root_dir "$CHECKPOINT_DIR"

echo ""
echo "Training complete. Checkpoints saved to ${CHECKPOINT_DIR}/"
