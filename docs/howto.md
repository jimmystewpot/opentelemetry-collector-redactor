# How-To Guide: Building, Configuring, and Operating the In-Process ONNX Redactor

This guide provides practical instructions for developers and site reliability engineers (SREs) deploying, operating, and optimising the `onnxredactionprocessor` within an OpenTelemetry Collector distribution.

All conventions and terminology in this document follow Australian English spelling.

---

## Dedicated Platform & GPU Guides

For detailed operating system, hardware vectorisation, and GPU server acceleration runbooks, see the dedicated modular guides:

- **[Platform Decision Matrix & Landing Guide](howto/README.md)**: Architectural comparison, throughput thresholds, and hardware selection.
- **[macOS Deployment Guide](howto/macos.md)**: Apple Silicon (`arm64`) & Intel (`x86_64`), `@rpath`, SIP, and local `ocb` development.
- **[Linux x86_64 Production Guide](howto/linux-x86_64.md)**: AVX2/AVX-512 VNNI, `glibc` vs `musl` ABI, and hardened `systemd` service unit.
- **[Linux ARM64 Production Guide](howto/linux-arm64.md)**: AWS Graviton 2/3/4, Ampere Altra, ARM NEON/I8MM, Docker Buildx, and NUMA tuning.
- **[GPU Acceleration & Server Integration Guide](howto/gpu-acceleration.md)**: In-process CUDA/ROCm execution providers and distributed remote Triton Inference Server offloading.

---

## Table of Contents

1. [Local Development & Environment Setup](#1-local-development--environment-setup)
2. [Setting Up ONNX Runtime Native Libraries](#2-setting-up-onnx-runtime-native-libraries)
3. [Exporting and Quantising a Custom NER Model](#3-exporting-and-quantising-a-custom-ner-model)
4. [Building the Collector Distribution with OCB](#4-building-the-collector-distribution-with-ocb)
5. [Configuring OpenTelemetry Pipelines](#5-configuring-opentelemetry-pipelines)
6. [Tuning Performance and Memory Limits](#6-tuning-performance-and-memory-limits)
7. [Production Operations & Troubleshooting Runbook](#7-production-operations--troubleshooting-runbook)

---

## 1. Local Development & Environment Setup

### Prerequisites

* **Operating System:** macOS (Apple Silicon ARM64 or Intel x86_64) or Linux (Ubuntu 22.04+, Debian 12+, RHEL 9+ on `aarch64` or `x86_64`).
* **Go Compiler:** Go 1.27 or newer.
* **C Toolchain:** `clang` (Xcode command line tools on macOS) or `gcc` / `build-essential` (Linux).
* **Python:** Python 3.10+ (for model export and quantisation scripts).

Verify your local Go environment:

```bash
go version
```

---

## 2. Setting Up ONNX Runtime Native Libraries

The processor relies on the ONNX Runtime C API. Headers and shared libraries must be accessible during compilation (`CGO_CFLAGS`, `CGO_LDFLAGS`) and at runtime (`LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`).

### Automated Download via Makefile

Run the `download-ort` target from the repository root:

```bash
make download-ort
```

This target detects your platform (`Darwin-arm64`, `Linux-aarch64`, `Linux-x64`) and unpacks ONNX Runtime 1.31.0 into `.ort/`:

```text
.ort/
├── include/
│   ├── onnxruntime_c_api.h
│   └── ...
└── lib/
    ├── libonnxruntime.1.31.0.dylib (or .so)
    └── libonnxruntime.dylib (or .so)
```

### Configuring Environment Variables for Manual Builds

When running `go test`, `go build`, or IDE tooling manually outside the Makefile, export the following environment variables:

#### On macOS:

```bash
export CGO_CFLAGS="-I$(pwd)/.ort/include"
export CGO_LDFLAGS="-L$(pwd)/.ort/lib -lonnxruntime -Wl,-rpath,$(pwd)/.ort/lib"
export DYLD_LIBRARY_PATH="$(pwd)/.ort/lib:${DYLD_LIBRARY_PATH:-}"
```

#### On Linux:

```bash
export CGO_CFLAGS="-I$(pwd)/.ort/include"
export CGO_LDFLAGS="-L$(pwd)/.ort/lib -lonnxruntime -Wl,-rpath,$(pwd)/.ort/lib"
export LD_LIBRARY_PATH="$(pwd)/.ort/lib:${LD_LIBRARY_PATH:-}"
```

Run the unit test suite to verify the native bridge:

```bash
make test
```

---

## 3. Exporting and Quantising a Custom NER Model

The processor uses a standard BERT-style transformer trained on token classification with 19 BIO output classes (`O` + 9 `B-*` + 9 `I-*`).

### Using the Python Model Export Script

To convert a fine-tuned HuggingFace model into a dynamic INT8 quantised ONNX format optimised for ARM64 NEON execution:

1. Install Python dependencies:

   ```bash
   pip install torch transformers onnx onnxruntime
   ```

2. Run the export script:

   ```bash
   python3 scripts/export_and_quantize.py \
       --model ./my-finetuned-ner-model \
       --output-dir ./models/ner-int8 \
       --quantize int8
   ```

3. Compute the SHA-256 integrity checksums for operator verification:

   ```bash
   sha256sum ./models/ner-int8/model_int8.onnx
   sha256sum ./models/ner-int8/tokenizer.json
   ```

### Synthetic Model Generation for Testing

For unit testing or CI environments without GPU infrastructure, generate deterministic synthetic ONNX models using:

```bash
python3 scripts/generate_test_onnx.py
```

This creates `tiny_ner.onnx` and `slow_loop.onnx` under `processor/onnxredactionprocessor/testdata/`.

---

## 4. Building the Collector Distribution with OCB

The OpenTelemetry Collector Builder (`ocb`) compiles custom collector binaries containing only required components.

1. Download the `ocb` binary:

   ```bash
   curl -sSL -o /usr/local/bin/ocb \
       "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/cmd%2Fbuilder%2Fv0.163.2/ocb_0.163.2_darwin_arm64"
   chmod +x /usr/local/bin/ocb
   ```

   *(Replace `darwin_arm64` with `linux_arm64` or `linux_amd64` as required).*

2. Review `builder-config.yaml`:

   ```yaml
   dist:
     name: otelcol-onnxredactor
     description: "OpenTelemetry Collector with in-process ONNX NER Redactor"
     output_path: ./bin
     version: 0.1.0
     cgo_enabled: true

   processors:
     - gomod: github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor v0.1.0
       path: ./processor/onnxredactionprocessor
     - gomod: go.opentelemetry.io/collector/processor/batchprocessor v0.163.0
     - gomod: go.opentelemetry.io/collector/processor/memorylimiterprocessor v0.163.0

   receivers:
     - gomod: go.opentelemetry.io/collector/receiver/otlpreceiver v0.163.0

   exporters:
     - gomod: go.opentelemetry.io/collector/exporter/otlpexporter v0.163.0
     - gomod: go.opentelemetry.io/collector/exporter/debugexporter v0.163.0
   ```

3. Execute the build:

   ```bash
   ocb --config=builder-config.yaml
   ```

The compiled binary will be placed at `./bin/otelcol-onnxredactor`.

---

## 5. Configuring OpenTelemetry Pipelines

### Recommended Pipeline Topology

Always place `onnxredaction` **after** `memory_limiter` and **before** `batch`:

```text
otlp receiver ──▶ memory_limiter ──▶ onnxredaction ──▶ batch ──▶ otlp exporter
```

* **`memory_limiter` first:** Ensures sudden traffic spikes trigger backpressure before memory is allocated for string extraction or tokenisation.
* **`onnxredaction` before `batch`:** Performs in-place masking and metric collision merging prior to outbound batching. This ensures colliding metric series resulting from redacted labels are merged before exporter serialisation.

### Sample Production Configuration

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
    check_interval: 500ms
    limit_percentage: 75
    spike_limit_percentage: 20

  onnxredaction:
    model:
      path: /etc/otel/models/tiny_ner.onnx
      sha256: 7ef06dbfe51de3cd4cb08c2075a390b21ad261b28e7c7f9e36e6e66d2d7329a8
      tokenizer_path: /etc/otel/models/tokenizer.json
      tokenizer_sha256: 2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884
      session:
        intra_op_threads: 2
        concurrent_runs: 2
        allow_spinning: false

    inference:
      coalesce_window: 2ms
      coalesce_max_strings: 64
      max_tensor_batch_size: 32
      length_buckets: [32, 128, 256]
      window_tokens: 256
      window_overlap_tokens: 32
      max_payload_bytes: 16384
      confidence_threshold: 0.85
      timeout: 25ms
      warmup_on_start: true

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

    filtering:
      min_string_length: 4
      exclude_attributes:
        - "service.name"
        - "service.version"
        - "host.*"
        - "k8s.*"
        - "container.*"
        - "telemetry.sdk.*"
        - "os.*"

    on_error: drop
    dry_run: false

  batch:
    send_batch_size: 8192
    timeout: 200ms

exporters:
  otlp:
    endpoint: backend:4317

service:
  pipelines:
    logs:
      receivers: [otlp]
      processors: [memory_limiter, onnxredaction, batch]
      exporters: [otlp]
    traces:
      receivers: [otlp]
      processors: [memory_limiter, onnxredaction, batch]
      exporters: [otlp]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, onnxredaction, batch]
      exporters: [otlp]

  telemetry:
    metrics:
      address: 0.0.0.0:8888
```

---

## 6. Tuning Performance and Memory Limits

### Memory Budgets & GOMEMLIMIT

In containerised Kubernetes environments, set the `GOMEMLIMIT` environment variable to roughly 70% of the container cgroup limit. This reserves 30% for native CGo ONNX Runtime arena memory:

```yaml
env:
  - name: GOMEMLIMIT
    value: "140MiB" # For a 200MiB memory limit container
```

### ONNX Runtime Session Tuning

* `intra_op_threads`: Controls intra-operator parallelism. On 2-core edge nodes, set to `2`. On 8-core gateways, set to `4`. Avoid setting this higher than the number of physical cores allocated to the pod.
* `concurrent_runs`: Sized to the worker semaphore. Default `2`. Sets the maximum parallel inference calls. Setting this too high increases CGo lock contention.
* `allow_spinning: false`: Prevents worker threads from busy-waiting on idle queues, conserving host CPU power when telemetry traffic dips.

### Coalescing Window Calibration

* `coalesce_window`: Default `2ms`. Balances batching throughput against pipeline latency.
  * For low-latency microservice meshes: set to `1ms`.
  * For high-throughput background log shippers: set to `5ms`.
* `coalesce_max_strings`: Default `64`. Dispatches the batch immediately once 64 candidate strings are accumulated, without waiting for the timer to expire.

---

## 7. Production Operations & Troubleshooting Runbook

### Alerting Recommendations

Set Prometheus alerts on the following key metrics:

1. **High Redaction Errors:**
   ```promql
   rate(redaction_errors_total[2m]) > 0.05
   ```
   *Action:* Check `reason` label. If `semaphore_full`, increase `concurrent_runs` or scale collector replicas. If `detector_error`, check model path or logs for CGo runtime errors.

2. **Inference Queuing Backlog:**
   ```promql
   histogram_quantile(0.99, rate(redaction_queue_wait_seconds_bucket[5m])) > 0.050
   ```
   *Action:* Indicates requests are waiting longer than 50 ms in the coalescing window. CPU starvation is occurring; increase CPU limits.

3. **Producer PII Leaks in Metrics:**
   ```promql
   sum by (metric) (rate(redaction_metric_label_pii_total[5m])) > 0
   ```
   *Action:* Alerts platform teams that an application metric (e.g. `http.requests`) is emitting PII inside its label keys/values, causing label scrubbing and collision merging.

### Runbook: Common Incidents

#### Issue A: `redaction_errors_total{reason="semaphore_full"}` Increasing

* **Symptom:** Upstream clients receive gRPC retryable backpressure errors; `redaction_errors_total` increments.
* **Root Cause:** Ingestion rate exceeds model inference throughput.
* **Resolution:**
  1. Verify if `min_string_length` can be raised from 4 to 8 to bypass trivial strings.
  2. Add high-cardinality non-PII attributes to `exclude_attributes` (e.g. `trace_id`, `span_id`).
  3. Increase pod replicas in the Horizontal Pod Autoscaler (HPA).

#### Issue B: Model Checksum Mismatch at Startup

* **Symptom:** Collector fails to start with:
  ```text
  ort: checksum mismatch for model /etc/otel/models/tiny_ner.onnx: expected <sha>, got <sha>
  ```
* **Root Cause:** The model file was modified, corrupted during download, or the configuration SHA-256 hash was not updated following model retraining.
* **Resolution:** Recompute `sha256sum /etc/otel/models/tiny_ner.onnx` and verify the integrity of the file before updating `config.yaml`.

#### Issue C: High In-Flight Metric Collisions

* **Symptom:** `redaction_metric_collisions_total` shows high merge rates.
* **Root Cause:** A metric contains high-cardinality identifiers (e.g. `account_id` or `username`) that were masked into `[REDACTED_PERSON]`, collapsing previously separate time series into one.
* **Resolution:** Investigate the producer service using `redaction_metric_label_pii_total{metric="<name>"}`. Update the application to avoid attaching high-cardinality PII as metric dimensions.
