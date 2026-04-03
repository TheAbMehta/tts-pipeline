import argparse
import json
import time
import wave

import numpy as np
import sherpa_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="test_sherpa.wav")
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    sample_rate = config.get("audio", {}).get("sample_rate", 22050)

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=args.model,
                tokens=args.config,
                data_dir="",
                dict_dir="",
            ),
            provider="cpu",
            num_threads=4,
        ),
    )

    print(f"Loading Sherpa-ONNX model: {args.model}")
    tts = sherpa_onnx.OfflineTts(tts_config)

    print(f"Synthesizing: {args.text}")
    start = time.perf_counter()
    audio = tts.generate(args.text, sid=args.sid, speed=args.speed)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if audio.sample_rate != sample_rate:
        print(f"  Note: Sherpa reports {audio.sample_rate}Hz (config says {sample_rate}Hz)")
        sample_rate = audio.sample_rate

    samples = np.array(audio.samples, dtype=np.float32)
    duration_s = len(samples) / sample_rate
    rtf = (elapsed_ms / 1000) / duration_s if duration_s > 0 else float("inf")

    print(f"  Audio: {duration_s:.2f}s at {sample_rate}Hz")
    print(f"  Inference: {elapsed_ms:.0f}ms (RTF: {rtf:.3f})")

    samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(args.output, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())

    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
