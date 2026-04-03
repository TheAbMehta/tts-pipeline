#!/usr/bin/env bash
set -euo pipefail

WORK=/home/abmehta/tts-pipeline
LOG=/var/log/tts-setup.log

exec > >(tee -a "$LOG") 2>&1
echo "=== TTS Pipeline Setup started at $(date) ==="

if ! nvidia-smi &>/dev/null; then
    echo "Installing NVIDIA drivers..."
    /opt/deeplearning/install-driver.sh || true
fi

apt-get update && apt-get install -y espeak-ng libespeak-ng-dev sox

pip install --no-cache-dir \
    piper-phonemize>=1.1.0 \
    onnxruntime-gpu>=1.16.0 \
    sherpa-onnx>=1.8.0 \
    'numpy<2' \
    tensorboard

pip install --no-cache-dir \
    'piper-tts[train] @ git+https://github.com/rhasspy/piper.git@2023.11.14-2#subdirectory=src/python' \
    || {
        pip install --no-cache-dir \
            git+https://github.com/rhasspy/piper.git@2023.11.14-2#subdirectory=src/python_run
        pip install --no-cache-dir \
            git+https://github.com/rhasspy/piper.git@2023.11.14-2#subdirectory=src/python_train
    }

cd "$WORK"
bash scripts/01_download_data.sh

bash scripts/02_preprocess.sh

echo "=== Setup complete at $(date). Ready to train. ==="
echo "READY" > /tmp/tts-setup-done
