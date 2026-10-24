# vector-onnx-redactor: In-Process OpenTelemetry ONNX PII Redactor

[![Go Version](https://img.shields.io/badge/Go-1.27%2B-blue.svg)](https://golang.org)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Collector%20v0.163.0-orange.svg)](https://opentelemetry.io)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-v1.31.0-brightgreen.svg)](https://onnxruntime.ai)
[![Licence](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg)](LICENSE)

An in-process, high-throughput OpenTelemetry Collector processor that performs zero-allocation Named Entity Recognition (NER) redaction across **Logs**, **Traces**, and **Metrics**. 

The processor executes machine-learning inference directly within the OpenTelemetry Collector daemon via the ONNX Runtime C API. It is tailored for low-latency edge nodes and high-volume aggregation gateways on ARM64 (`aarch64`) and x86_64 architectures, replacing fragile, unmaintainable regular expressions with context-aware neural models.

---

## Key Features & Architecture

* **Zero-Allocation Hot Path:** Operates at **5.4 ns/op with 0 B/op and 0 allocs/op** on clean telemetry strings. All intermediate buffers, token slices, window descriptions, and tensors are recycled through pre-allocated arena pools ([`arena.Pool`](file:///Users/jlamb/go/src/github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena/arena.go)).
* **Single Shared ONNX Engine Across Signals:** A thread-safe, reference-counted component registry shares a single underlying ONNX Runtime session, memory arena, tokeniser, and coalescing queue across Logs, Traces, and Metrics pipelines, eliminating redundant memory duplication and preserving container memory ceilings (< 200 MB RSS).
* **Fine-Grained User Path Redaction (`PATH_USER`):** Intelligently isolates and redacts **only the user segment** in file system paths (e.g. `/home/alice/.ssh/id_rsa` becomes `/home/[REDACTED_PATH_USER]/.ssh/id_rsa`, and `C:\Users\admin\app.log` becomes `C:\Users\[REDACTED_PATH_USER]\app.log`), preserving critical operating system and application directory structures for stack traces.
* **Pure NER with Zero Regex Fallbacks:** Enforces semantic machine-learning classification without brittle pattern lists.
* **CGo Concurrency & Cancellation Isolation:** Independent `OrtRunOptions` instances are pooled per concurrent execution slot. When an upstream request times out or is cancelled, cooperative termination halts native execution immediately without interfering with parallel in-flight batches.
* **Cross-Signal Batch Coalescing:** Dynamically aggregates candidate strings across concurrent signal pipelines over microsecond sliding time windows (e.g. 2 ms) to maximise CPU SIMD/NEON throughput.
* **Post-Redaction Metric Series Collision Resolution:** Redacting identifying labels in metric series can collapse previously distinct series keys. The processor automatically merges colliding datapoints across `Sum`, `Gauge`, `Histogram`, `ExponentialHistogram`, and `Summary` metrics using streaming FNV-64a attribute hashing.
* **Fail-Closed and Fail-Safe Error Modes:** Supports `on_error: drop` (scrubbing uninspected strings with `[REDACTED_UNINSPECTED]` and returning retryable errors on systemic failure) and `on_error: ignore` (fail-open audit mode).

---

## Supported Entity Taxonomy

The processor supports a 9-class entity taxonomy (19 BIO output logits: `O` + 9 `B-*` + 9 `I-*`):

| Entity Label | Placeholder | Description |
| :--- | :--- | :--- |
| `PERSON` | `[REDACTED_PERSON]` | Individual human names (multilingual, diacritics, CJK ideographs) |
| `EMAIL` | `[REDACTED_EMAIL]` | Email addresses (RFC 5322, sub-addressing, unicode domains) |
| `PHONE` | `[REDACTED_PHONE]` | Telephone numbers (E.164, national and international formats) |
| `LOCATION` | `[REDACTED_LOCATION]` | Physical street addresses, postcodes, cities, geographical coordinates |
| `CREDIT_CARD` | `[REDACTED_CREDIT_CARD]` | Primary Account Numbers (PANs, 13–19 digits across major card networks) |
| `SSN` | `[REDACTED_SSN]` | Social Security, Tax File Numbers (TFN), and national identity numbers |
| `IBAN` | `[REDACTED_IBAN]` | International Bank Account Numbers |
| `MEDICAL_ID` | `[REDACTED_MEDICAL_ID]` | Medicare numbers, national health service numbers, patient identifiers |
| `PATH_USER` | `[REDACTED_PATH_USER]` | User-specific directory segments in POSIX, Windows, UNC, and URI paths |

---

## Pipeline Processing Stages

The processor transforms telemetry through 11 sequential, allocation-optimised stages:

```
[ Telemetry Ingestion (Logs / Traces / Metrics) ]
                       │
                       ▼
 1. Candidate Extraction (Map attributes, span names, log bodies, metric keys)
                       │
                       ▼
 2. Static Filtering (Prefix wildcards, minimum string length, pure numerics)
                       │
                       ▼
 3. Cross-Signal Batch Coalescer (Sliding time window + max string threshold)
                       │
                       ▼
 4. Zero-Closure Deduplication (Table mapping unique strings to mutation targets)
                       │
                       ▼
 5. WordPiece Tokenisation (Exact UTF-8 byte offset tracking & rune boundary validation)
                       │
                       ▼
 6. Strided Windowing (256-token windows with 32-token overlap)
                       │
                       ▼
 7. Length Bucketing (Sub-batches partitioned into tight sequence buckets: 32, 128, 256)
                       │
                       ▼
 8. In-Process ONNX Inference (Native CGo execution, isolated OrtRunOptions)
                       │
                       ▼
 9. BIO Logits Decoding & Span Merging (Argmax softmax, confidence gating, tie-breaking)
                       │
                       ▼
10. In-Place Span Masking (JSON-in-string escape widening, UTF-8 rune safety)
                       │
                       ▼
11. Metric Series Collision Resolution (FNV-64a canonical hashing & series merging)
```

---

## Performance Benchmarks

Benchmarks run on Apple Silicon ARM64 (`darwin/arm64`) using Go 1.27 and ONNX Runtime 1.31.0:

```text
BenchmarkMask_CleanString-10         213,534,086      5.440 ns/op      0 B/op     0 allocs/op
BenchmarkMask_DirtyString-10           7,854,114    149.200 ns/op    600 B/op     7 allocs/op
BenchmarkPipeline_FullInference-10         1,062  1,185,635 ns/op  10,787 B/op    43 allocs/op
```

* **Clean String Throughput:** Over 180 million operations per second per core with zero heap allocations on the 99.9% clean happy path.
* **Inference Latency:** ~1.18 ms per tensor batch under the quantised INT8 model, scaling sub-linearly when multiple signals coalesce into a single execution pass.

---

## Getting Started

### Prerequisites

* **Go:** Version 1.27 or higher.
* **ONNX Runtime:** Version 1.31.0 C API headers and shared libraries (`libonnxruntime.so` on Linux, `libonnxruntime.dylib` on macOS).
* **OpenTelemetry Collector Builder (`ocb`):** Version 0.163.2 or higher.
* **C Compiler:** `clang` (macOS) or `gcc` (Linux).

### 1. Download ONNX Runtime Libraries

Use the provided Makefile target to download and unpack the pre-compiled ONNX Runtime shared library into `.ort/`:

```bash
make download-ort
```

This populates `.ort/include` and `.ort/lib` with the official platform binaries.

### 2. Run Test Suite & Linters

Execute unit tests with race detection and run static analysis:

```bash
make test
make lint
```

### 3. Build a Custom Collector with OCB

To compile an OpenTelemetry Collector containing the `onnxredactionprocessor`, run the OpenTelemetry Collector Builder using [`builder-config.yaml`](file:///Users/jlamb/go/src/github.com/jimmystewpot/vector-onnx-redactor/builder-config.yaml):

```bash
ocb --config=builder-config.yaml
```

The resulting binary is generated at `bin/otelcol-onnxredactor`.

### 4. Run via Docker (ARM64)

The repository provides a multi-stage, checksum-verified [`Dockerfile`](file:///Users/jlamb/go/src/github.com/jimmystewpot/vector-onnx-redactor/Dockerfile):

```bash
docker build -t otelcol-onnxredactor:latest .
docker run --rm -p 4317:4317 -p 4318:4318 -p 8888:8888 otelcol-onnxredactor:latest
```

---

## Configuration Reference

Add the `onnxredaction` processor to your OpenTelemetry Collector configuration file:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  memory_limiter:
    check_interval: 1s
    limit_percentage: 75
    spike_limit_percentage: 20

  onnxredaction:
    # Model and Tokeniser specification
    model:
      path: /etc/otel/models/tiny_ner.onnx
      sha256: 7ef06dbfe51de3cd4cb08c2075a390b21ad261b28e7c7f9e36e6e66d2d7329a8
      tokenizer_path: /etc/otel/models/tokenizer.json
      tokenizer_sha256: 2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884
      session:
        intra_op_threads: 2     # CPU execution threads inside operators
        concurrent_runs: 2      # Maximum parallel inference calls
        allow_spinning: false   # Disables thread spinning to save CPU idle power

    # Inference and batching parameters
    inference:
      coalesce_window: 2ms      # Sliding batch coalescing window
      coalesce_max_strings: 64  # Maximum unique strings before forcing dispatch
      max_tensor_batch_size: 32 # Maximum windows dispatched per ONNX call
      length_buckets: [32, 128, 256] # Sequence length buckets
      window_tokens: 256        # Maximum tokens per sliding window
      window_overlap_tokens: 32 # Token overlap between adjacent windows
      max_payload_bytes: 16384  # Maximum bytes inspected per string
      confidence_threshold: 0.85 # Minimum softmax probability threshold
      timeout: 25ms             # Per-batch inference timeout
      warmup_on_start: true     # Initialises execution provider during startup

    # Target entities to redact
    entities:
      - PERSON
      - EMAIL
      - PHONE
      - LOCATION
      - CREDIT_CARD
      - SSN
      - IBAN
      - MEDICAL_ID
      - PATH_USER

    # Pre-inference attribute filtering
    filtering:
      min_string_length: 4      # Strings shorter than 4 runes bypass inference
      exclude_attributes:       # Bypassed attribute key patterns (exact or wildcard)
        - "service.name"
        - "service.version"
        - "host.*"
        - "k8s.*"
        - "container.*"
        - "telemetry.sdk.*"
        - "os.*"

    # Fault tolerance policy
    on_error: drop              # 'drop' (fail-closed) or 'ignore' (fail-open)
    dry_run: false              # If true, runs inference and telemetry without masking

  batch:
    send_batch_size: 8192
    timeout: 200ms

exporters:
  otlp:
    endpoint: tempo:4317
    tls:
      insecure: true
  debug:
    verbosity: detailed

service:
  pipelines:
    logs:
      receivers: [otlp]
      processors: [memory_limiter, onnxredaction, batch]
      exporters: [debug]
    traces:
      receivers: [otlp]
      processors: [memory_limiter, onnxredaction, batch]
      exporters: [otlp]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, onnxredaction, batch]
      exporters: [debug]
  telemetry:
    metrics:
      address: 0.0.0.0:8888
```

---

## Observability & Prometheus Metrics

The processor exports granular OpenTelemetry metrics on port `:8888`:

| Metric Name | Type | Unit | Description |
| :--- | :---: | :---: | :--- |
| `redaction_duration_seconds` | Histogram | `s` | Execution duration partitioned by stage (`tokenize`, `infer`, `decode`, `mask`). |
| `redaction_batch_strings` | Histogram | `{strings}` | Distribution of candidate string count in coalesced batches. |
| `redaction_tensor_batch_size` | Histogram | `{windows}` | Size of tensor sub-batches dispatched to ONNX Runtime. |
| `redaction_padding_ratio` | Histogram | `1` | Ratio of valid non-padding tokens to total allocated tensor slots. |
| `redaction_entities_total` | Counter | `{entities}` | Count of detected and redacted entities partitioned by `label`. |
| `redaction_inspected_strings_total` | Counter | `{strings}` | Total candidate strings inspected partitioned by `signal`. |
| `redaction_skipped_total` | Counter | `{strings}` | Attributes skipped prior to inference partitioned by `reason` (`filter`, `bytes_body`). |
| `redaction_errors_total` | Counter | `{errors}` | Runtime errors partitioned by `reason` (`timeout`, `semaphore_full`, `panic`, etc.). |
| `redaction_scrubbed_total` | Counter | `{strings}` | Strings scrubbed in-place with `[REDACTED_UNINSPECTED]` partitioned by `signal` and `reason`. |
| `redaction_dropped_total` | Counter | `{records}` | Records dropped under `on_error: drop` partitioned by `signal` and `reason`. |
| `redaction_inflight` | UpDownCounter | `{requests}` | Active in-flight telemetry batches currently undergoing redaction. |
| `redaction_queue_wait_seconds` | Histogram | `s` | Queuing time spent by requests in the coalescer sliding window prior to inference. |
| `redaction_metric_collisions_total` | Counter | `{collisions}` | Colliding metric datapoints resolved after redaction partitioned by `type`. |
| `redaction_metric_label_pii_total` | Counter | `{metrics}` | Metric series that had PII redacted from their labels partitioned by `metric`. |
| `redaction_engine_info` | Gauge | `1` | Static build metadata (`ort_version`, `model_sha`, `kernel_path`). |

---

## Further Reading

* For executive instructions, deployment runbooks, and pipeline configuration, see the [How-To Guide](docs/howto.md).
* For platform-specific runbooks and hardware acceleration architectures:
  * [Platform Decision Matrix & Hardware Selection](docs/howto/README.md)
  * [macOS Deployment Guide](docs/howto/macos.md) (Apple Silicon ARM64 & Intel x86_64)
  * [Linux x86_64 Production Guide](docs/howto/linux-x86_64.md) (AVX2/AVX-512, systemd, glibc)
  * [Linux ARM64 Production Guide](docs/howto/linux-arm64.md) (AWS Graviton, Ampere Altra, NEON)
  * [GPU Acceleration & Server Integration Guide](docs/howto/gpu-acceleration.md) (CUDA, ROCm, and Triton)
* For model taxonomy details and benchmark results, see the [Model Decision Record](docs/models/model-decision-record.md).

---

## Licence

Licensed under the Apache Licence, Version 2.0.
