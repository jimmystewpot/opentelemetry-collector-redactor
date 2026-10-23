#!/usr/bin/env python3
"""
Model export and quantization pipeline for onnxredactionprocessor.

Exports a HuggingFace NER transformer model to ONNX and quantizes weights
to INT8 using ONNX Runtime dynamic quantization, targeting ARM64 NEON
(AWS Graviton 2/3 and Apple Silicon).

Usage:
    python3 scripts/export_and_quantize.py \
        --model dslim/bert-base-NER \
        --output-dir models/bert-base-ner-int8 \
        --quantize int8
"""

import argparse
import os
import sys
import json
import numpy as np

def export_model(model_name: str, output_path: str, opset: int = 17):
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError:
        print("Error: torch and transformers required for model export.", file=sys.stderr)
        print("Install via: pip install torch transformers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dummy_input = tokenizer("John Doe lives in New York.", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]

    print(f"Exporting to ONNX at {output_path} (opset {opset})...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Export complete: {output_path}")

    # Save tokenizer JSON alongside model
    tokenizer_json_path = os.path.join(os.path.dirname(output_path), "tokenizer.json")
    tokenizer.save_pretrained(os.path.dirname(output_path))
    print(f"Saved tokenizer files to: {os.path.dirname(output_path)}")

def quantize_model(input_path: str, output_path: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("Error: onnxruntime required for quantization.", file=sys.stderr)
        print("Install via: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    print(f"Quantizing {input_path} to INT8 -> {output_path}...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Quantization complete: {orig_size:.1f} MB -> {quant_size:.1f} MB ({quant_size/orig_size*100:.1f}%)")

def verify_model(model_path: str):
    import onnxruntime as ort

    print(f"Verifying {model_path} with ONNX Runtime...")
    sess = ort.InferenceSession(model_path)
    inputs = {inp.name: inp for inp in sess.get_inputs()}
    outputs = {out.name: out for out in sess.get_outputs()}

    assert "input_ids" in inputs, "Missing input_ids"
    assert "attention_mask" in inputs, "Missing attention_mask"
    assert "logits" in outputs, "Missing logits"

    # Test dummy inference
    b, s = 1, 16
    feed = {
        "input_ids": np.ones((b, s), dtype=np.int64),
        "attention_mask": np.ones((b, s), dtype=np.int64),
    }
    res = sess.run(None, feed)
    assert res[0].shape[:2] == (b, s), f"Unexpected shape {res[0].shape}"
    print(f"Verification successful: output shape = {res[0].shape}")

def main():
    parser = argparse.ArgumentParser(description="Export and quantize NER models for OpenTelemetry Collector.")
    parser.add_argument("--model", type=str, default="dslim/bert-base-NER", help="HuggingFace model ID or path")
    parser.add_argument("--output-dir", type=str, default="models/exported", help="Output directory")
    parser.add_argument("--quantize", choices=["none", "int8"], default="int8", help="Quantization mode")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify exported model with ONNX Runtime")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fp32_path = os.path.join(args.output_dir, "model_fp32.onnx")
    int8_path = os.path.join(args.output_dir, "model_int8.onnx")

    export_model(args.model, fp32_path)

    if args.quantize == "int8":
        quantize_model(fp32_path, int8_path)
        final_model = int8_path
    else:
        final_model = fp32_path

    if args.verify:
        verify_model(final_model)

if __name__ == "__main__":
    main()
