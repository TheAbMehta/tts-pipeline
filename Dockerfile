FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PIPER_COMMIT=2023.11.14-2

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libespeak-ng-dev \
    git \
    wget \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install piper-phonemize (pre-built wheel)
RUN pip install --no-cache-dir \
    piper-phonemize>=1.1.0 \
    onnxruntime-gpu>=1.16.0 \
    sherpa-onnx>=1.8.0 \
    'numpy<2' \
    tensorboard

# Install piper_train from source (pinned commit)
RUN pip install --no-cache-dir \
    'piper-tts[train] @ git+https://github.com/rhasspy/piper.git@${PIPER_COMMIT}#subdirectory=src/python' \
    || pip install --no-cache-dir \
    git+https://github.com/rhasspy/piper.git@${PIPER_COMMIT}#subdirectory=src/python_run \
    && pip install --no-cache-dir \
    git+https://github.com/rhasspy/piper.git@${PIPER_COMMIT}#subdirectory=src/python_train

WORKDIR /workspace/tts-pipeline

COPY . .

CMD ["bash"]
