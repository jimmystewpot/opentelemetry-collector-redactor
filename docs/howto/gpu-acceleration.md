# GPU Acceleration & Server Integration Guide

This guide details how to integrate and operate GPU hardware acceleration with the OpenTelemetry Collector and `onnxredactionprocessor`. It covers two distinct architectural patterns:

- **Pattern A (In-Process GPU Acceleration):** Compiling and executing ONNX Runtime native GPU execution providers (NVIDIA CUDA/TensorRT and AMD ROCm/MIGraphX) directly within the Collector process.
- **Pattern B (Distributed Remote Inference Offload):** Decoupling telemetry collection from neural compute by streaming batched token tensors over high-throughput gRPC to a centralised NVIDIA Triton Inference Server cluster.

All conventions and terminology in this document follow Australian English.

---

## 1. Architectural Patterns Overview

```
                                [ Incoming Telemetry Stream ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
     [ Pattern A: In-Process GPU ]                   [ Pattern B: Remote Triton Offload ]
  ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
  │ OpenTelemetry Collector Gateway     │         │ Lightweight Edge Collector Pod      │
  │  - Attribute Extraction             │         │  - Attribute Extraction             │
  │  - WordPiece Tokenisation           │         │  - WordPiece Tokenisation           │
  │  - In-Process CUDA / ROCm Provider  │         │  - Batch Coalescer                  │
  │    (Direct CGo VRAM Buffers)        │         └──────────────────┬──────────────────┘
  │  - In-Place String Masking          │                            │ (gRPC Stream)
  │  - Series Collision Resolution      │                            ▼
  └─────────────────────────────────────┘         ┌─────────────────────────────────────┐
                                                  │ Centralised GPU Server (Triton)     │
                                                  │  - Dynamic Batch Scheduler (2ms)    │
                                                  │  - Multi-Instance GPU Pool (A10G)   │
                                                  │  - TensorRT Optimised ONNX Engine   │
                                                  └──────────────────┬──────────────────┘
                                                                     │ (Logits Response)
                                                                     ▼
                                                  ┌─────────────────────────────────────┐
                                                  │ Edge Collector Post-Processing      │
                                                  │  - Span Decoding & Masking          │
                                                  │  - Metric Collision Resolution      │
                                                  └─────────────────────────────────────┘
```

---

## 2. Pattern A: In-Process GPU Acceleration (Local CGo Providers)

In Pattern A, the Collector daemon links directly to GPU-accelerated ONNX Runtime provider libraries. Tensors reside in GPU device memory (VRAM), eliminating network hops and delivering sub-500 microsecond inference latencies.

### 2.1 NVIDIA CUDA & TensorRT

#### Prerequisites
- **Host GPU:** NVIDIA Turing, Ampere, Ada Lovelace, or Hopper architecture (e.g. T4, L4, A10G, A100).
- **Host Driver:** NVIDIA Linux Driver version 550.x or higher.
- **CUDA Toolkit & cuDNN:** CUDA 12.x and cuDNN 9.x runtime libraries.
- **NVIDIA Container Toolkit:** `nvidia-container-toolkit` installed on Docker/Kubernetes container hosts.

#### Dynamic Library Dependencies
The GPU-enabled ONNX Runtime distribution (`onnxruntime-linux-x64-gpu-1.31.0.tgz`) includes:
- `libonnxruntime.so`
- `libonnxruntime_providers_shared.so`
- `libonnxruntime_providers_cuda.so`
- `libonnxruntime_providers_tensorrt.so`

#### CGo Engine Extension Blueprint
To register the CUDA execution provider in `ort_shim.c`, use the official ONNX Runtime C API:

```c
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int ort_shim_enable_cuda(OrtSessionOptions* opts, int device_id, char** out_err) {
    OrtCUDAProviderOptionsV2* cuda_opts = NULL;
    OrtStatus* st = g_ort->CreateCUDAProviderOptions(&cuda_opts);
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return -1;
    }

    // Configure dedicated GPU VRAM memory arena ceiling (2 GiB)
    char device_str[16];
    snprintf(device_str, sizeof(device_str), "%d", device_id);

    const char* keys[] = {
        "device_id",
        "arena_extend_strategy",
        "gpu_mem_limit",
        "cudnn_conv_algo_search"
    };
    const char* values[] = {
        device_str,
        "kSameAsRequested",
        "2147483648", // 2 GiB VRAM limit
        "DEFAULT"
    };
    size_t num_keys = sizeof(keys) / sizeof(keys[0]);

    st = g_ort->UpdateCUDAProviderOptions(cuda_opts, keys, values, num_keys);
    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        g_ort->ReleaseCUDAProviderOptions(cuda_opts);
        return -1;
    }

    st = g_ort->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cuda_opts);
    g_ort->ReleaseCUDAProviderOptions(cuda_opts);

    if (st) {
        if (out_err) *out_err = strdup(g_ort->GetErrorMessage(st));
        g_ort->ReleaseStatus(st);
        return -1;
    }
    return 0;
}
```

#### Multi-Stage GPU Dockerfile (`Dockerfile.cuda`)

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.27-bookworm AS builder

ARG ORT_VERSION=1.31.0
ARG OCB_VERSION=0.163.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tar build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN curl -sSL -o onnxruntime-gpu.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz" \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf onnxruntime-gpu.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm onnxruntime-gpu.tgz

RUN curl -sSL -o /usr/local/bin/ocb \
    "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/cmd%2Fbuilder%2Fv${OCB_VERSION}/ocb_${OCB_VERSION}_linux_amd64" \
    && chmod +x /usr/local/bin/ocb

WORKDIR /src
COPY . .

ENV CGO_CFLAGS="-I/opt/onnxruntime/include"
ENV CGO_LDFLAGS="-L/opt/onnxruntime/lib -lonnxruntime -Wl,-rpath,/usr/local/lib"

RUN ocb --config=builder-config.yaml

# Runtime stage with NVIDIA CUDA 12.4
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libcudnn9-cuda-12 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/onnxruntime/lib/libonnxruntime*.so* /usr/local/lib/
RUN ldconfig

RUN groupadd -g 10001 otel && useradd -u 10001 -g otel -s /bin/false otel \
    && mkdir -p /etc/otel/models /var/log/otel \
    && chown -R otel:otel /etc/otel /var/log/otel

COPY --from=builder /src/bin/otelcol-onnxredactor /usr/local/bin/otelcol
COPY --from=builder /src/config.yaml /etc/otel/config.yaml
COPY --from=builder /src/processor/onnxredactionprocessor/testdata/tiny_ner.onnx /etc/otel/models/tiny_ner.onnx
COPY --from=builder /src/processor/onnxredactionprocessor/testdata/tokenizer.json /etc/otel/models/tokenizer.json

USER 10001:10001
EXPOSE 4317 4318 8888
ENTRYPOINT ["/usr/local/bin/otelcol", "--config=/etc/otel/config.yaml"]
```

#### Running Container with GPU Passthrough
```bash
docker run -d \
  --gpus all \
  --name otelcol-gpu \
  -p 4317:4317 -p 4318:4318 -p 8888:8888 \
  otelcol-onnxredactor:cuda
```

---

### 2.2 AMD ROCm & MIGraphX

#### Prerequisites
- **Hardware:** AMD Instinct MI100, MI210, MI250, MI300 or Radeon Pro GPUs.
- **Host Drivers:** ROCm 6.1+ kernel drivers (`amdgpu-dkms`).
- **Device Access:** Permissions to access `/dev/kfd` (Kernel Fusion Driver) and `/dev/dri` (Direct Rendering Infrastructure).

#### Container Execution on AMD ROCm
Launch the container forwarding ROCm devices and user groups:
```bash
docker run -d \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --name otelcol-rocm \
  -p 4317:4317 -p 4318:4318 \
  otelcol-onnxredactor:rocm
```

---

### 2.3 Kubernetes GPU Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otelcol-gpu-gateway
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: otelcol-gpu-gateway
  template:
    metadata:
      labels:
        app: otelcol-gpu-gateway
    spec:
      containers:
        - name: otelcol
          image: myregistry/otelcol-onnxredactor:cuda
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: 4Gi
              cpu: "4000m"
            requests:
              nvidia.com/gpu: 1
              memory: 2Gi
              cpu: "2000m"
          ports:
            - containerPort: 4317
            - containerPort: 4318
            - containerPort: 8888
```

---

## 3. Pattern B: Distributed Remote Inference Offload (Triton Architecture)

For large Kubernetes clusters with hundreds of lightweight microservice pods, deploying GPUs on every telemetry node is cost-prohibitive. Pattern B centralises GPU hardware into a shared NVIDIA Triton Inference Server pool.

### 3.1 Triton Model Repository Layout

```text
model_repository/
└── ner_redactor/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

### 3.2 Triton Model Configuration (`config.pbtxt`)

```protobuf
name: "ner_redactor"
platform: "onnxruntime_onnx"
max_batch_size: 128

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, 19 ]
  }
]

# Dynamic batch scheduler coalesces concurrent client requests across a 2ms window
dynamic_batching {
  max_queue_delay_microseconds: 2000
  preferred_batch_size: [ 32, 64, 128 ]
}

# Deploy 2 model execution instances per GPU
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# Optimisation parameters
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt"
      parameters { key: "precision_mode" value: "INT8" }
      parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    }]
  }
}
```

### 3.3 Starting Triton Inference Server
```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.04-py3 \
  tritonserver --model-repository=/models
```

---

## 4. Latency, Throughput & Cost Trade-Off Analysis

| Metric / Dimension | In-Process CPU SIMD | Pattern A: In-Process GPU | Pattern B: Remote Triton Gateway |
| :--- | :--- | :--- | :--- |
| **Inference Latency (p99)** | ~1.18 ms | ~0.45 ms | ~2.50 ms (1.0 ms network + 0.3 ms GPU + 1.2 ms batching) |
| **Network Overhead** | 0 bytes (in-process memory) | 0 bytes (host-to-device PCIe) | gRPC tensor serialisation over cluster network |
| **Collector RAM Footprint**| < 200 MB RSS | < 350 MB RSS (+ 2 GB VRAM) | < 80 MB RSS |
| **Throughput per Node** | 10k to 25k strings/sec | 60k to 120k strings/sec | Edge: 30k strings/sec; Triton: 250k+ strings/sec |
| **Failure Domain** | Isolated to single collector pod | Isolated to single collector pod | Shared inference dependency across collector fleet |
| **Cost Profile** | Lowest (Standard compute) | Highest per collector pod | Highly efficient amortised GPU utilisation |

---

## 5. Resilience & Operational Troubleshooting

### GPU Out-of-Memory (VRAM OOM)
- **Symptom:** ONNX Runtime reports `CUDA failure 2: out of memory`.
- **Mitigation:**
  1. Enforce `gpu_mem_limit` in session options to prevent unbounded dynamic allocation.
  2. Reduce `max_tensor_batch_size` in `config.yaml` from 64 to 32.
  3. Ensure `on_error: drop` is configured so that uninspected telemetry strings are scrubbed with `[REDACTED_UNINSPECTED]` and upstream backpressure is communicated cleanly.

### CUDA Context Resets & Thermal Throttling
- When deploying high-density aggregation nodes, monitor GPU temperature and throttled clock events via `nvidia-smi dmon`.
- In Kubernetes, configure liveness probes inspecting `/metrics` to automatically recycle unhealthy pods experiencing hardware stalls.
