import argparse
import os
import shutil

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_int8(input_path: str, output_path: str) -> None:
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )


def convert_fp16(input_path: str, output_path: str) -> None:
    from onnxconverter_common import float16

    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    name = args.model_name

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Input model: {input_path} ({input_size:.1f} MB)")

    int8_path = os.path.join(output_dir, f"{name}-int8.onnx")
    print(f"\nQuantizing to INT8: {int8_path}")
    quantize_int8(input_path, int8_path)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"  INT8 model: {int8_size:.1f} MB ({int8_size/input_size:.1%} of original)")

    config_src = input_path + ".json"
    if os.path.exists(config_src):
        shutil.copy2(config_src, int8_path + ".json")

    try:
        fp16_path = os.path.join(output_dir, f"{name}-fp16.onnx")
        print(f"\nConverting to FP16: {fp16_path}")
        convert_fp16(input_path, fp16_path)
        fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
        print(f"  FP16 model: {fp16_size:.1f} MB ({fp16_size/input_size:.1%} of original)")
        if os.path.exists(config_src):
            shutil.copy2(config_src, fp16_path + ".json")
    except ImportError:
        print("\nSkipping FP16 (install onnxconverter-common for FP16 support)")
    except Exception as e:
        print(f"\nFP16 conversion failed: {e}")

    print("\nQuantization complete.")


if __name__ == "__main__":
    main()
