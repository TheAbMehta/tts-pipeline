#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="data"
LJSPEECH_URL="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJSPEECH_DIR="${DATA_DIR}/LJSpeech-1.1"

mkdir -p "$DATA_DIR"

if [ -d "$LJSPEECH_DIR" ]; then
    echo "LJSpeech already downloaded at ${LJSPEECH_DIR}"
else
    echo "Downloading LJSpeech-1.1 (~2.6GB)..."
    wget -c "$LJSPEECH_URL" -O "${DATA_DIR}/LJSpeech-1.1.tar.bz2"

    echo "Extracting..."
    tar xjf "${DATA_DIR}/LJSpeech-1.1.tar.bz2" -C "$DATA_DIR"

    echo "Cleaning up archive..."
    rm -f "${DATA_DIR}/LJSpeech-1.1.tar.bz2"
fi

if [ -d "${LJSPEECH_DIR}/wavs" ] && [ ! -e "${LJSPEECH_DIR}/wav" ]; then
    echo "Creating wav/ symlink for Piper compatibility..."
    ln -s wavs "${LJSPEECH_DIR}/wav"
fi

echo "Converting metadata to Piper format..."
awk -F'|' '{print $1 "|" $3}' "${LJSPEECH_DIR}/metadata.csv" > "${LJSPEECH_DIR}/metadata_piper.csv"

LINES=$(wc -l < "${LJSPEECH_DIR}/metadata_piper.csv")
echo "Done. ${LINES} utterances ready (expected ~13100)."
