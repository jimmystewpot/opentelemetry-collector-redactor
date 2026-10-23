#!/usr/bin/env python3
"""
Generate deterministic synthetic ONNX test fixtures for onnxredactionprocessor.
- tiny_ner.onnx: Fast Gather+Add model returning logits int64[B,S] -> float32[B,S,19]
- slow_loop.onnx: Long-running Loop model to test cooperative cancellation via RunOptionsSetTerminate
- tokenizer.json: Bert-style WordPiece tokenizer fixture with standard special tokens
"""

import json
import os
import hashlib
import numpy as np
import onnx
from onnx import helper, TensorProto

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "processor", "onnxredactionprocessor", "testdata"
)

VOCAB_SIZE = 1000
NUM_CLASSES = 19  # 0: O, 1..9: B-*, 10..18: I-*

LABELS = [
    "O",
    "B-PERSON", "B-EMAIL", "B-PHONE", "B-LOCATION", "B-CREDIT_CARD", "B-SSN", "B-IBAN", "B-MEDICAL_ID", "B-PATH_USER",
    "I-PERSON", "I-EMAIL", "I-PHONE", "I-LOCATION", "I-CREDIT_CARD", "I-SSN", "I-IBAN", "I-MEDICAL_ID", "I-PATH_USER",
]

def make_tiny_ner_model():
    # input_ids: [batch_size, seq_len] (int64)
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "seq_len"])
    # attention_mask: [batch_size, seq_len] (int64)
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "seq_len"])

    # logits: [batch_size, seq_len, 19] (float32)
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "seq_len", NUM_CLASSES])

    # Embedding table W: [VOCAB_SIZE, NUM_CLASSES]
    # Initialize with default class 0 ("O") having high logit (+10.0), others (-10.0)
    w_data = np.full((VOCAB_SIZE, NUM_CLASSES), -10.0, dtype=np.float32)
    w_data[:, 0] = 10.0

    # Specific deterministic mappings for testing entity detection:
    # Token IDs 200..208 -> B-PERSON .. B-PATH_USER
    # Token IDs 210..218 -> I-PERSON .. I-PATH_USER
    for i in range(1, 10):
        w_data[200 + i - 1, :] = -10.0
        w_data[200 + i - 1, i] = 10.0
    for i in range(1, 10):
        w_data[210 + i - 1, :] = -10.0
        w_data[210 + i - 1, i + 9] = 10.0

    w_init = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[VOCAB_SIZE, NUM_CLASSES],
        vals=w_data.flatten().tolist()
    )

    zero_val = helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0])

    axes_2 = helper.make_tensor("axes_2", TensorProto.INT64, [1], [2])

    nodes = [
        # Gather embedding from W: output shape [batch_size, seq_len, NUM_CLASSES]
        helper.make_node("Gather", inputs=["W", "input_ids"], outputs=["raw_logits"], axis=0),
        # Ensure attention_mask is consumed in computation without affecting logits
        helper.make_node("Cast", inputs=["attention_mask"], outputs=["mask_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", inputs=["mask_float", "zero_val"], outputs=["mask_zero"]),
        # Unsqueeze mask_zero to [batch_size, seq_len, 1] using axes tensor for opset 13+
        helper.make_node("Unsqueeze", inputs=["mask_zero", "axes_2"], outputs=["mask_unsq"]),
        # Add mask_unsq to raw_logits -> logits
        helper.make_node("Add", inputs=["raw_logits", "mask_unsq"], outputs=["logits"]),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="TinyNER",
        inputs=[input_ids, attention_mask],
        outputs=[logits],
        initializer=[w_init, zero_val, axes_2]
    )

    model = helper.make_model(
        graph,
        producer_name="onnxredactionprocessor-test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model)
    return model

def make_slow_loop_model():
    # A model designed to take measurable time and allow cooperative cancellation
    # It contains a Loop node with 200,000 iterations of vector arithmetic
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "seq_len"])
    attention_mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch_size", "seq_len"])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", "seq_len", NUM_CLASSES])

    # Base Gather graph first
    w_data = np.zeros((VOCAB_SIZE, NUM_CLASSES), dtype=np.float32)
    w_init = helper.make_tensor("W", TensorProto.FLOAT, [VOCAB_SIZE, NUM_CLASSES], w_data.flatten().tolist())

    # Loop configuration:
    # max_trip_count: 500,000 iterations
    trip_count = helper.make_tensor("trip_count", TensorProto.INT64, [], [500000])
    cond_true = helper.make_tensor("cond_true", TensorProto.BOOL, [], [True])

    # Loop state variable: tensor of shape [256, 256]
    state_init = helper.make_tensor("state_init", TensorProto.FLOAT, [256, 256], np.eye(256, dtype=np.float32).flatten().tolist())

    # Subgraph for Loop
    # Inputs: iter_num (int64), cond_in (bool), state_in (float[256, 256])
    iter_num = helper.make_tensor_value_info("iter_num", TensorProto.INT64, [])
    cond_in = helper.make_tensor_value_info("cond_in", TensorProto.BOOL, [])
    state_in = helper.make_tensor_value_info("state_in", TensorProto.FLOAT, [256, 256])

    # Outputs: cond_out (bool), state_out (float[256, 256])
    cond_out = helper.make_tensor_value_info("cond_out", TensorProto.BOOL, [])
    state_out = helper.make_tensor_value_info("state_out", TensorProto.FLOAT, [256, 256])

    # Loop body: state_out = MatMul(state_in, state_in) * 0.9999 + 0.0001
    const_factor = helper.make_tensor("const_factor", TensorProto.FLOAT, [1], [0.99999])
    subgraph_nodes = [
        helper.make_node("Identity", inputs=["cond_in"], outputs=["cond_out"]),
        helper.make_node("MatMul", inputs=["state_in", "state_in"], outputs=["mm"]),
        helper.make_node("Mul", inputs=["mm", "const_factor"], outputs=["state_out"]),
    ]

    subgraph = helper.make_graph(
        nodes=subgraph_nodes,
        name="SlowLoopBody",
        inputs=[iter_num, cond_in, state_in],
        outputs=[cond_out, state_out],
        initializer=[const_factor]
    )

    loop_node = helper.make_node(
        "Loop",
        inputs=["trip_count", "cond_true", "state_init"],
        outputs=["state_final"],
        body=subgraph
    )

    axes_2 = helper.make_tensor("axes_2", TensorProto.INT64, [1], [2])
    zero_const = helper.make_tensor("zero_const", TensorProto.FLOAT, [1], [0.0])

    clean_nodes = [
        helper.make_node("Gather", inputs=["W", "input_ids"], outputs=["raw_logits"], axis=0),
        loop_node,
        helper.make_node("ReduceSum", inputs=["state_final"], outputs=["state_sum"], keepdims=0),
        helper.make_node("Mul", inputs=["state_sum", "zero_const"], outputs=["zero_scalar"]),
        helper.make_node("Cast", inputs=["attention_mask"], outputs=["mask_float"], to=TensorProto.FLOAT),
        helper.make_node("Mul", inputs=["mask_float", "zero_const"], outputs=["mask_zero"]),
        helper.make_node("Unsqueeze", inputs=["mask_zero", "axes_2"], outputs=["mask_unsq"]),
        helper.make_node("Add", inputs=["raw_logits", "mask_unsq"], outputs=["logits_temp"]),
        helper.make_node("Add", inputs=["logits_temp", "zero_scalar"], outputs=["logits"]),
    ]

    graph = helper.make_graph(
        nodes=clean_nodes,
        name="SlowLoopNER",
        inputs=[input_ids, attention_mask],
        outputs=[logits],
        initializer=[w_init, trip_count, cond_true, state_init, zero_const, axes_2]
    )

    model = helper.make_model(
        graph,
        producer_name="onnxredactionprocessor-test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model)
    return model

def make_tokenizer_json():
    vocab = {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
        "[MASK]": 103,
    }
    # Deterministic entity tokens for testing:
    # 200..208: B-*, 210..218: I-*
    for i, label in enumerate(LABELS[1:10]):
        vocab[f"token_{label.lower()}"] = 200 + i
    for i, label in enumerate(LABELS[10:19]):
        vocab[f"token_{label.lower()}"] = 210 + i

    # Common vocabulary words for testing redaction & tokenization
    common_words = [
        "the", "a", "an", "is", "in", "at", "to", "for", "from", "with", "on", "by", "of",
        "user", "alice", "bob", "john", "doe", "smith", "jane",
        "home", "var", "log", "tmp", "etc", "opt", "usr", "bin",
        "id_rsa", "password", "secret", "token", "key",
        "email", "gmail", "com", "example", "org", "net",
        "call", "request", "response", "status", "code", "error", "failed", "success",
        "http", "https", "get", "post", "path", "api", "v1", "v2",
        "card", "credit", "ssn", "iban", "phone",
        "##alice", "##bob", "##smith", "##doe", "##com", "##org", "##net",
    ]
    next_id = 300
    for w in common_words:
        if w not in vocab:
            vocab[w] = next_id
            next_id += 1

    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 100, "content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 101, "content": "[CLS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 102, "content": "[SEP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 103, "content": "[MASK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True
        },
        "pre_tokenizer": {
            "type": "BertPreTokenizer"
        },
        "post_processor": {
            "type": "BertProcessing",
            "sep": ["[SEP]", 102],
            "cls": ["[CLS]", 101]
        },
        "decoder": {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": True
        },
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab
        }
    }
    return tok

def main():
    os.makedirs(TESTDATA_DIR, exist_ok=True)

    tiny_ner = make_tiny_ner_model()
    tiny_ner_path = os.path.join(TESTDATA_DIR, "tiny_ner.onnx")
    onnx.save(tiny_ner, tiny_ner_path)
    print(f"Generated {tiny_ner_path}")

    slow_loop = make_slow_loop_model()
    slow_loop_path = os.path.join(TESTDATA_DIR, "slow_loop.onnx")
    onnx.save(slow_loop, slow_loop_path)
    print(f"Generated {slow_loop_path}")

    tok_data = make_tokenizer_json()
    tok_path = os.path.join(TESTDATA_DIR, "tokenizer.json")
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, indent=2)
    print(f"Generated {tok_path}")

    # Compute sha256 checksums
    checksums = []
    for fname in sorted(["tiny_ner.onnx", "slow_loop.onnx", "tokenizer.json"]):
        p = os.path.join(TESTDATA_DIR, fname)
        with open(p, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        checksums.append(f"{digest}  {fname}")

    checksum_path = os.path.join(TESTDATA_DIR, "checksums.txt")
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(checksums) + "\n")
    print(f"Recorded checksums in {checksum_path}")

if __name__ == "__main__":
    main()
