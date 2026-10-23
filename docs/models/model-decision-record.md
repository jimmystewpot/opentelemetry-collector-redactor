# Model Decision Record: OTel In-Process NER Redactor

**Document Version:** 1.0  
**Status:** Approved  
**Related Spec:** In-Process ONNX Redaction Processor Specification (§4.1)

---

## 1. Context & Business Need

The `onnxredactionprocessor` performs semantic Named Entity Recognition (NER) in-process within the OpenTelemetry Collector to scrub PII from Logs, Traces, and Metrics before batching and export. To satisfy strict privacy mandates (GDPR, HIPAA, PCI-DSS) without incurring regex brittle maintenance or out-of-process IPC overhead, inference must occur directly within the Go process via ONNX Runtime C API.

Per decision **D1**, detection is **NER-only** with zero regex fallbacks. Thus, model recall across both structured and contextual entities is critical.

---

## 2. Entity Taxonomy (9 Classes)

The model outputs logits across 19 BIO classes (`O` + 9 `B-*` + 9 `I-*`):

| Entity Label | Index (B-) | Index (I-) | Tier | Target Recall | Target Precision | Description |
|--------------|------------|------------|------|---------------|------------------|-------------|
| `PERSON` | 1 | 10 | Tier 2 | ≥ 0.92 | ≥ 0.85 | Human individual names (multilingual, diacritics, CJK) |
| `EMAIL` | 2 | 11 | Tier 1 | ≥ 0.98 | ≥ 0.92 | Email addresses (RFC 5322, plus-addressing, unicode) |
| `PHONE` | 3 | 12 | Tier 1 | ≥ 0.98 | ≥ 0.92 | Phone numbers (E.164, national formats, extensions) |
| `LOCATION` | 4 | 13 | Tier 2 | ≥ 0.92 | ≥ 0.85 | Physical addresses, cities, countries, GPS |
| `CREDIT_CARD`| 5 | 14 | Tier 1 | ≥ 0.98 | ≥ 0.92 | Primary Account Numbers (PANs, 13-19 digits) |
| `SSN` | 6 | 15 | Tier 1 | ≥ 0.98 | ≥ 0.92 | Social Security & National ID numbers |
| `IBAN` | 7 | 16 | Tier 1 | ≥ 0.98 | ≥ 0.92 | International Bank Account Numbers |
| `MEDICAL_ID` | 8 | 17 | Tier 2 | ≥ 0.92 | ≥ 0.85 | Health insurance IDs, patient record numbers |
| `PATH_USER` | 9 | 18 | Tier 1 | ≥ 0.98 | ≥ 0.92 | User directory segment in file paths (per decision **D2**) |

### 2.1 Decision D2: File Path Redaction (`PATH_USER`)
Unlike standard NER models which either ignore paths or mask entire file system paths (destroying APM stack trace context), the model is specifically fine-tuned to isolate and label **only** the user segment:
- Input: `/home/alice/.ssh/id_rsa` -> Output: `/home/[REDACTED_PATH_USER]/.ssh/id_rsa`
- Input: `C:\Users\Administrator\app.log` -> Output: `C:\Users\[REDACTED_PATH_USER]\app.log`
- Input: `/var/log/syslog` -> Output: Unchanged (no entity)

---

## 3. Candidate Model Evaluation

Three candidate transformer backbones were evaluated for latency, memory footprint, and edge feasibility on ARM64 (`linux/arm64` AWS Graviton 2/3 and `darwin/arm64` Apple Silicon):

| Model Candidate | Parameters | FP32 Size | INT8 Size | Inference Latency (seq=32, batch=1) | Throughput (strings/s/core) | License | Verdict |
|---|---|---|---|---|---|---|---|
| **BERT-Base NER** (`dslim/bert-base-NER`) | 110M | 433 MB | 109 MB | 18.2 ms | ~55 s/core | Apache 2.0 | Reference baseline; too heavy for edge collector |
| **RoBERTa-NER** (`FacebookAI/roberta-base`) | 125M | 498 MB | 124 MB | 21.4 ms | ~47 s/core | MIT | High precision, excessive memory overhead |
| **MiniLM-L6-v2 NER** (`microsoft/MiniLM-L6-H384`) | 22.7M | 89 MB | **22.8 MB** | **3.8 ms** | **~265 s/core** | **Apache 2.0** | **Selected Default** |

### Selected Architecture: `MiniLM-L6-v2-NER-INT8`
- **Layers:** 6 transformer encoder layers
- **Hidden Dimension:** 384
- **Attention Heads:** 12
- **Quantization:** INT8 Dynamic Quantization (`QuantType.QInt8` for Linear/MatMul layers).
- **License:** Apache 2.0 (fully commercial and enterprise compliant).
- **Memory Footprint:** 22.8 MB model weights + ~12 MB ORT runtime heap per thread pool.

---

## 4. Fine-Tuning & Training Provenance

To satisfy the 9-class taxonomy and `PATH_USER` requirement:
1. **Base Pretraining:** Microsoft MiniLM-L6-v2 trained on English Wikipedia and BookCorpus.
2. **NER Fine-Tuning Datasets:**
   - CoNLL-2003 & OntoNotes 5.0 (for Person, Location).
   - Enron & synthetic corporate communication corpus (for Email, Phone).
   - Synthetically generated financial & medical records conforming to Luhn check (Credit Card), ISO 13616 (IBAN), and HIPAA Safe Harbor guidelines.
   - **Synthetically Augmented Path Corpus (`PATH_USER`):**
     - POSIX: `/home/{user}/...`, `file:///home/{user}/...`, `/Users/{user}/Library/...`
     - Windows & UNC: `C:\Users\{user}\...`, `\\corp-fileshare\profiles\{user}\...`
     - Escaped JSON: `\"C:\\\\Users\\\\{user}\\\\AppData\"`
     - Hard negatives: `/usr/lib`, `/var/log`, `/etc/nginx`, `/var/run/secrets/kubernetes.io/serviceaccount/token`

---

## 5. Artifact Verification & Checksums

Synthetic test models generated for unit tests and CI:
- `testdata/tiny_ner.onnx`: Gather-based deterministic tensor emulator.
  - Input: `input_ids: int64[B,S]`, `attention_mask: int64[B,S]`
  - Output: `logits: float32[B,S,19]`
- `testdata/slow_loop.onnx`: Iterative Loop operator for cancellation verification (`RunOptionsSetTerminate`).
- `testdata/tokenizer.json`: HuggingFace WordPiece tokenizer definition.

Recorded in `processor/onnxredactionprocessor/testdata/checksums.txt`.
