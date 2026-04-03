# TTS Pipeline: Train a Voice, Run It Anywhere

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19426354.svg)](https://doi.org/10.5281/zenodo.19426354)
![Zenodo Views](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fzenodo.org%2Fapi%2Frecords%2F19426354&query=%24.stats.version_views&label=views&color=blue)
![Zenodo Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fzenodo.org%2Fapi%2Frecords%2F19426354&query=%24.stats.version_downloads&label=downloads&color=green)

A complete pipeline for training a neural text-to-speech voice from scratch and deploying it in a web browser, on a phone, or on a Raspberry Pi. No cloud API calls at inference time. Your model, your hardware, your data.

Built on [VITS](https://arxiv.org/abs/2106.06103) (via [Piper](https://github.com/rhasspy/piper)) and [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) for cross-platform inference.

## What This Actually Does

You give it audio + transcripts. It gives you a tiny model (~20 MB) that speaks in that voice, entirely offline. The whole thing trains in about 3 hours on two A100 GPUs and costs roughly $14 in GCP spot pricing. Not free, but not grad-school-grant money either.

The trained model runs in:
- **Web browsers** via WebAssembly (yes, really, no server)
- **Android / iOS** via Sherpa-ONNX mobile libraries
- **Raspberry Pi** and other ARM boards
- **Any desktop** with ONNX Runtime

## Quick Start

```bash
# 1. Spin up a GPU instance (or use your own)
gcloud compute instances create tts-train \
  --machine-type=a2-highgpu-2g \
  --zone=us-central1-a \
  --accelerator=count=2,type=nvidia-tesla-a100 \
  --provisioning-model=SPOT \
  --metadata-from-file=startup-script=gcp-startup.sh

# 2. SSH in and kick off training
gcloud compute ssh tts-train
cd ~/tts-pipeline
bash scripts/03_train.sh

# 3. Export + quantize when it finishes
bash scripts/04_export_onnx.sh
bash scripts/05_quantize.sh

# 4. Test it
bash scripts/06_test_inference.sh
```

Or if you just want to try the web demo with the pre-trained model:

```bash
cd web/
python3 serve.py
# Open http://localhost:8080
```

## Project Layout

```
tts-pipeline/
  scripts/
    01_download_data.sh      # Grab LJSpeech (2.6 GB)
    02_preprocess.sh         # Phonemize + spectrograms
    03_train.sh              # Train VITS on GPU(s)
    04_export_onnx.sh        # Checkpoint -> ONNX
    05_quantize.sh           # INT8 + FP16 variants
    06_test_inference.sh     # Sanity check
  src/
    quantize_model.py        # INT8/FP16 quantization logic
  web/
    index.html               # Browser demo UI
    app.js                   # TTS logic, audio playback
    serve.py                 # Dev server (COOP/COEP headers)
    sherpa-onnx-*.js/wasm    # Sherpa-ONNX WASM runtime
  output/
    en-us-ljspeech-medium.onnx       # Full FP32 model (~61 MB)
    en-us-ljspeech-medium.onnx.json  # Config (phoneme map, sample rate, etc.)
  docs/
    training_report.tex      # LaTeX write-up with loss curves
  gcp-startup.sh             # Automated GCP VM provisioning
  Dockerfile                 # Container-based training (alternative)
```

## The Pipeline, Step by Step

### 1. Data (LJSpeech)

13,100 clips of a single English speaker reading public-domain books. About 24 hours of audio, 22050 Hz, 16-bit. It is the "hello world" of TTS datasets, well understood and freely available.

### 2. Preprocessing

eSpeak-NG converts text to IPA phonemes. Mel spectrograms get precomputed and cached. This takes maybe 10 minutes and produces a `training/` directory with everything the model needs.

### 3. Training

VITS is a strange and beautiful model. It is a VAE, a normalizing flow, and a GAN all at the same time. The generator learns to produce raw audio waveforms directly from phoneme sequences. The discriminator tries to tell real audio from fake. They fight, and good speech comes out.

We use Piper's training wrapper (`piper_train`) on PyTorch Lightning. Two A100 GPUs, batch size 128 effective, 10,000 steps. The loss starts around 60 and settles near 43. The discriminator loss oscillates like adversarial losses always do. This is normal. If it stops oscillating, something is wrong.

Training takes roughly 3 hours. You will see "audio amplitude out of range, auto clipped" warnings constantly. These are normal. The model is learning.

### 4. Export

`piper_train.export_onnx` converts the PyTorch checkpoint to ONNX. The result is a ~61 MB file that can run on any ONNX Runtime. Two gotchas here:

**PyTorch 2.6+ serialization.** `torch.load` now defaults to `weights_only=True`, which rejects Piper checkpoints because they contain `pathlib.PosixPath` objects. Fix: call `torch.serialization.add_safe_globals([pathlib.PosixPath])` before loading.

**Missing ONNX metadata.** This one bit us hard. Piper's export does not add the metadata fields that Sherpa-ONNX needs to run inference. Without them, the model loads fine but crashes at phonemization time with "Failed to set eSpeak-ng voice." The fix is to patch the ONNX file after export:

```python
import onnx
model = onnx.load("model.onnx")
for key, value in [
    ("sample_rate", "22050"),
    ("add_blank", "1"),
    ("n_speakers", "1"),
    ("voice", "en-us"),       # must match an espeak-ng voice name
    ("comment", "piper"),     # tells sherpa-onnx to use the Piper frontend
    ("language", "English"),
]:
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value
onnx.save(model, "model.onnx")
```

The `voice` field is the one that matters most. Sherpa-ONNX passes it directly to eSpeak-NG as the voice identifier. If it is missing or wrong, eSpeak-NG cannot initialize and you get silent failure in the browser or a crash in native builds. We spent a while debugging this because the WASM version fails silently while the native version at least throws an error message.

### 5. Quantization

Dynamic INT8 quantization shrinks the model from 61 MB to ~20 MB. For TTS specifically, INT8 works surprisingly well because the decoder (HiFi-GAN) is robust to small weight perturbations. We also produce an FP16 variant (~30 MB) as a fallback.

### 6. Web Deployment

The ONNX model gets bundled into a Sherpa-ONNX WASM build. The browser downloads the model once, caches it, and runs inference entirely client-side. No server. No API key. No latency penalty from network round trips.

The web demo is a single page: type text, click Speak, hear audio. Speed slider, generation time display, the basics. Ctrl+Enter to speak without clicking.

## What Went Well

- **The pipeline is genuinely end-to-end.** Raw audio goes in, a browser-ready model comes out. Every step is scripted and reproducible.
- **Cost is reasonable.** $14 for a custom voice model is pretty good. Spot instances make A100s accessible to individuals, not just companies.
- **Browser deployment works.** Hearing your trained model speak inside a web page, with no server involved, is a satisfying moment. The latency is acceptable for most uses.
- **VITS quality at 10K steps is decent.** Not production-grade, but clearly intelligible and recognizably in the target voice. More steps would help.
- **The tooling ecosystem is solid.** Piper for training, Sherpa-ONNX for inference, eSpeak-NG for phonemization. These projects fit together well.

## What Went Poorly (Honest Version)

- **Dependency hell.** Piper pins `torch<2` but modern GCP VMs ship PyTorch 2.7. Lightning 1.7.7 doesn't like PyTorch 2.7's scheduler API. CUDA libraries exist in five different locations and none of them are on LD_LIBRARY_PATH by default. Getting the stack to cooperate took longer than the actual training.
- **The WASM .data file is huge.** The pre-built English model bundles to ~93 MB. That is a lot to ask a user to download on first visit. You can optimize this (strip unused espeak-ng languages, quantize the model first), but out of the box it is chunky.
- **No streaming.** VITS generates the entire utterance at once. For long text, there is a noticeable wait before audio starts. Chunking the text helps, but it introduces prosody breaks at chunk boundaries.
- **Model swapping requires a full WASM rebuild.** The model is baked into the .data file at compile time via Emscripten's `--preload-file`. You cannot just drop in a new .onnx file. Each model needs its own WASM build. This is the biggest friction point for iteration.
- **10,000 steps is probably not enough.** The loss was still trending down when training stopped. 20K-50K steps would likely produce noticeably better audio. We chose 10K to keep costs and iteration time down, but quality suffers.
- **Spot instance risk is real.** We got lucky and the instance survived the full training run. On a bad day, GCP will preempt you mid-training and you lose ~18 minutes of work (the checkpoint interval). Not catastrophic, but annoying.

## Compatibility Notes

This pipeline was built and tested with:
- Python 3.10 (3.14 breaks piper-train's Cython extensions)
- PyTorch 2.7.1+cu126
- PyTorch Lightning 1.7.7 (with a small scheduler validation patch)
- NVIDIA A100 GPUs (CUDA 12.8)
- Emscripten 3.1.x for WASM builds

Other configurations might work. They also might not. The dependency situation in the Piper/PyTorch/Lightning triangle is fragile.

## Hardware Requirements

**For training:**
- Minimum: 1x GPU with 16 GB VRAM (reduce batch size to 32)
- Recommended: 2x A100 40 GB (what we used)
- Training time scales roughly linearly with GPU count

**For inference:**
- Any modern CPU (x86 or ARM)
- ~20 MB RAM for the INT8 model
- A web browser from this decade (for the WASM demo)

## References

- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103) (Kim et al., ICML 2021)
- [Piper TTS](https://github.com/rhasspy/piper) (Rhasspy project)
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) (k2-fsa, next-gen Kaldi)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/) (Keith Ito, 2017)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646) (Kong et al., NeurIPS 2020)

## License

The code in this repo is MIT. The LJSpeech dataset is public domain. Piper is MIT. Sherpa-ONNX is Apache 2.0. The trained model weights inherit from LJSpeech (public domain) and Piper (MIT).
