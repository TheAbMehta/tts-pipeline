import argparse
import json
import time
import wave

import numpy as np
import onnxruntime as ort


def phonemize_text(text: str, config: dict) -> list[int]:
    from piper_phonemize import phonemize_espeak

    phoneme_id_map = config["phoneme_id_map"]
    language = config.get("espeak", {}).get("voice", "en-us")

    phoneme_strs = phonemize_espeak(text, language)

    phoneme_ids = [config.get("phoneme_id_map", {}).get("^", [0])[0]]
    if "^" in phoneme_id_map:
        phoneme_ids = list(phoneme_id_map["^"])

    for sentence_phonemes in phoneme_strs:
        for phoneme in sentence_phonemes:
            if phoneme in phoneme_id_map:
                phoneme_ids.extend(phoneme_id_map[phoneme])
                if "_" in phoneme_id_map:
                    phoneme_ids.extend(phoneme_id_map["_"])

    if "$" in phoneme_id_map:
        phoneme_ids.extend(phoneme_id_map["$"])

    return phoneme_ids


def synthesize(session: ort.InferenceSession, phoneme_ids: list[int],
               config: dict) -> np.ndarray:
    ids = np.array([phoneme_ids], dtype=np.int64)
    lengths = np.array([len(phoneme_ids)], dtype=np.int64)

    inference_cfg = config.get("inference", {})
    noise_scale = np.array([inference_cfg.get("noise_scale", 0.667)], dtype=np.float32)
    length_scale = np.array([inference_cfg.get("length_scale", 1.0)], dtype=np.float32)
    noise_w = np.array([inference_cfg.get("noise_w", 0.8)], dtype=np.float32)

    input_names = [inp.name for inp in session.get_inputs()]
    feed = {"input": ids, "input_lengths": lengths}

    if "scales" in input_names:
        scales = np.array([[noise_scale[0], length_scale[0], noise_w[0]]], dtype=np.float32)
        feed["scales"] = scales
    else:
        if "noise_scale" in input_names:
            feed["noise_scale"] = noise_scale
        if "length_scale" in input_names:
            feed["length_scale"] = length_scale
        if "noise_w" in input_names:
            feed["noise_w"] = noise_w

    if "sid" in input_names:
        feed["sid"] = np.array([0], dtype=np.int64)

    start = time.perf_counter()
    audio = session.run(None, feed)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    audio = audio.squeeze()
    return audio, elapsed_ms


def save_wav(audio: np.ndarray, path: str, sample_rate: int = 22050) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="test_output.wav")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    sample_rate = config.get("audio", {}).get("sample_rate", 22050)

    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    print(f"Phonemizing: {args.text}")
    phoneme_ids = phonemize_text(args.text, config)
    print(f"  {len(phoneme_ids)} phoneme IDs")

    print("Synthesizing...")
    audio, elapsed_ms = synthesize(session, phoneme_ids, config)

    duration_s = len(audio) / sample_rate
    rtf = (elapsed_ms / 1000) / duration_s if duration_s > 0 else float("inf")

    print(f"  Audio: {duration_s:.2f}s at {sample_rate}Hz")
    print(f"  Inference: {elapsed_ms:.0f}ms (RTF: {rtf:.3f})")

    save_wav(audio, args.output, sample_rate)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
