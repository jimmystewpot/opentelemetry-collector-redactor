# Design Specification: Platform & GPU Integration HOWTO Documentation Suite

**Date:** 2026-10-24  
**Status:** Approved  
**Language Standard:** Australian English (en-AU)  
**Target Repository:** `vector-onnx-redactor`  

---

## 1. Executive Summary & Problem Context

The `vector-onnx-redactor` project provides a zero-allocation, high-throughput OpenTelemetry Collector processor (`onnxredactionprocessor`) performing in-process Named Entity Recognition (NER) redaction on telemetry data. The processor relies heavily on native CGo bindings communicating with Microsoft's ONNX Runtime C API (`onnxruntime_c_api.h`).

Operating native CGo binaries across heterogeneous production platforms introduces non-trivial complexities:
1. **Dynamic Linking & RPATH:** macOS Mach-O binaries resolve dylibs using `@rpath` and are subject to System Integrity Protection (SIP) clearing `DYLD_LIBRARY_PATH`, whereas Linux ELF binaries rely on `DT_RPATH`, `DT_RUNPATH`, or `ldconfig`.
2. **CPU Instruction Extensions:** Modern transformer token classification performance heavily depends on hardware SIMD capabilities—specifically Intel/AMD AVX2 and AVX-512 VNNI vs ARM NEON and I8MM on 64-bit ARM platforms.
3. **C Library ABIs:** Pre-compiled ONNX Runtime Linux releases are built against GNU `glibc`. Running inside lightweight Alpine Linux (`musl`) environments causes dynamic linker failures or silent segmentation faults.
4. **GPU Acceleration Pathways:** Deploying neural redaction at enterprise telemetry volumes (tens of thousands of spans/logs per second) demands clear architectural guidance on when to run in-process GPU execution providers (NVIDIA CUDA / TensorRT, AMD ROCm) versus offloading inference to dedicated remote inference clusters (such as Triton Inference Server).

This specification defines the complete information architecture, technical contents, and build asset enhancements to establish a modular, production-grade HOWTO documentation suite.

---

## 2. Information Architecture & Repository Layout

The documentation is modularised into dedicated, standalone guides under `docs/howto/`, coordinated by a master decision matrix and landing index. The legacy root guide (`docs/howto.md`) is preserved as a concise executive overview and navigational router to maintain backwards compatibility with existing hyperlinks.

```text
docs/
├── howto.md                                   # Root overview & router pointing to docs/howto/
└── howto/
    ├── README.md                              # Master index, platform matrix & decision tree
    ├── macos.md                               # macOS workstation & edge server guide (ARM64 / x86_64)
    ├── linux-x86_64.md                        # Enterprise Linux x86_64 production guide (AVX2/512, systemd, OCI)
    ├── linux-arm64.md                         # Enterprise Linux ARM64 production guide (Graviton, Ampere, NEON)
    └── gpu-acceleration.md                    # In-process CUDA/ROCm & Triton remote offloading guide
```

---

## 3. Platform Decision Matrix & Selection Guidance

`docs/howto/README.md` serves as the initial architectural entry point for platform engineers and SREs. It provides a structured decision framework comparing deployment models:

| Platform / Topology | Primary Compute | Recommended Use Case | p99 Latency (Batch of 32) | Memory Overhead | Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **macOS (Apple Silicon)** | Apple M-series CPU (NEON) | Local development, testing, developer workstations | ~1.18 ms | < 200 MB RSS | Low |
| **macOS (Intel x86_64)** | Intel Core / Xeon (AVX2) | Legacy workstation development | ~1.85 ms | < 200 MB RSS | Low |
| **Linux x86_64** | Intel Xeon / AMD EPYC (AVX-512) | Central aggregation gateways, on-prem Kubernetes clusters | ~1.05 ms | < 250 MB RSS | Medium |
| **Linux ARM64** | AWS Graviton 3/4, Ampere Altra | Cloud-native edge collectors, cost-optimised Kubernetes clusters | ~1.12 ms | < 200 MB RSS | Medium |
| **In-Process GPU (CUDA)** | NVIDIA L4 / T4 / A10G | Extreme single-node telemetry aggregation (> 50k strings/s) | ~0.45 ms | > 1.5 GB VRAM | High |
| **In-Process GPU (ROCm)** | AMD Instinct MI210 / MI300 | Open-ecosystem high-volume telemetry clusters | ~0.50 ms | > 2.0 GB VRAM | High |
| **Remote Triton Gateway** | Dedicated GPU Server Cluster | Shared centralised inference across multi-tenant collectors | ~2.50 ms (incl. RTT) | Edge: < 80 MB | High |

### Architectural Decision Tree
- **Under 10,000 strings/sec per node:** Deploy CPU in-process inference using ARM64 (AWS Graviton) or x86_64 with AVX2/AVX-512. Sub-millisecond latency with zero GPU driver operational overhead.
- **10,000 – 50,000 strings/sec per node:** Coalesce larger batches (`coalesce_max_strings: 128`, `coalesce_window: 5ms`) on multi-core CPU nodes, scaling horizontally with Kubernetes Horizontal Pod Autoscaler (HPA).
- **Over 50,000 strings/sec per node or strict microsecond tensor deadlines:** Evaluate In-Process GPU acceleration (Pattern A) for dedicated gateway pods.
- **Heterogeneous / Multi-Tenant Fleet:** If edge collectors run in memory-constrained environments (< 150 MB RAM per container), evaluate Remote Triton Offload (Pattern B) to decouple inference compute from telemetry collection.

---

## 4. Platform Specifications

### 4.1 macOS Deployment (`docs/howto/macos.md`)
- **Supported Targets:** macOS Sonoma (14.x) and Sequoia (15.x) on Apple Silicon (`arm64`, M1 through M4) and Intel (`x86_64`).
- **Prerequisites:** Xcode Command Line Tools (`xcode-select --install`), Go 1.22+, OpenTelemetry Collector Builder (`ocb`), GNU Make, `curl`, `tar`.
- **Dynamic Linking & CGo Configuration:**
  - Mach-O dynamic linker semantics: Explain `-Wl,-rpath` usage.
  - Setting environment variables:
    ```bash
    export CGO_CFLAGS="-I$(pwd)/.ort/include"
    export CGO_LDFLAGS="-L$(pwd)/.ort/lib -lonnxruntime -Wl,-rpath,$(pwd)/.ort/lib"
    export DYLD_LIBRARY_PATH="$(pwd)/.ort/lib:${DYLD_LIBRARY_PATH:-}"
    ```
  - Apple System Integrity Protection (SIP) considerations: SIP clears `DYLD_LIBRARY_PATH` when spawning protected system binaries; explain why `-Wl,-rpath` baked into the binary is essential.
  - Ad-hoc code signing for local binaries: `codesign -s - --force ./bin/otelcol-onnxredactor`.
- **Hardware Acceleration Analysis:**
  - Standard CPU execution provider leverages ARM NEON vector instructions natively on Apple Silicon.
  - Evaluation of Apple CoreML Execution Provider: Why CPU INT8 quantisation remains optimal for sub-millisecond, low-payload telemetry strings compared to the compilation and tensor marshalling overhead of the Neural Engine on small batch sizes.
- **Recipes:** Complete step-by-step instructions to download ORT, compile with `ocb`, verify with `otlp-loadgen` / `curl`, and inspect active memory with `vmmap`.

---

### 4.2 Linux x86_64 Deployment (`docs/howto/linux-x86_64.md`)
- **Supported Distributions:** Debian 12 (Bookworm), Ubuntu 22.04/24.04 LTS, Red Hat Enterprise Linux (RHEL) 9, Rocky Linux 9.
- **Hardware & SIMD Optimisations:**
  - Verification of processor flags (`grep -m1 -E 'avx512|avx2|vnni' /proc/cpuinfo`).
  - Leveraging Intel Deep Learning Boost (VNNI) with dynamic INT8 quantised ONNX models.
- **C Library ABI & Linker Nuances:**
  - Detailed explanation of `glibc` dependency in official ONNX Runtime releases.
  - Caution against Alpine Linux (`musl`): Pre-built ONNX Runtime dynamic libraries fail against `musl-libc` with unresolved symbols (`__isnanf`, `gnu_get_libc_version`). Recommendations for Debian/Ubuntu slim base images.
  - Configuration of system-wide linker cache via `/etc/ld.so.conf.d/onnxruntime.conf` and `ldconfig`.
- **Production Host Deployment (systemd):**
  - Concrete `systemd` service unit definition (`/etc/systemd/system/otelcol-onnxredactor.service`) including:
    - Dedicated unprivileged service user (`otel`).
    - Cgroup memory limits (`MemoryHigh=180M`, `MemoryMax=220M`).
    - Go runtime memory limit (`Environment=GOMEMLIMIT=140MiB`).
    - Security hardening flags (`ProtectSystem=strict`, `ProtectHome=true`, `NoNewPrivileges=true`).
- **Containerisation (OCI / Docker):**
  - Standalone multi-stage Docker build targeting `linux/amd64`.
  - Non-root user execution, security contexts, and container health probes on port `:8888`.

---

### 4.3 Linux ARM64 Deployment (`docs/howto/linux-arm64.md`)
- **Supported Environments:** AWS Graviton 2/3/4 (`c7g`, `m7g`, `c8g`), Ampere Altra (Oracle OCI A1, Google Cloud Tau T2A), and edge server hardware (Raspberry Pi 5, NVIDIA Jetson AGX Orin in CPU mode).
- **ARM64 Architecture & Instruction Optimisations:**
  - ARMv8.2-A+ Dot Product (`asimddp`) and ARMv8.6-A Int8 Matrix Multiplication (`i8mm`) verification.
  - Performance impact on BERT tokeniser and INT8 tensor matrix multiplication.
- **Multi-Architecture Build Pipelines:**
  - Building ARM64 images natively or via Docker Buildx:
    ```bash
    docker buildx build --platform linux/arm64 -t otelcol-onnxredactor:arm64 --load .
    ```
  - Cross-compilation setup with Go `CGO_ENABLED=1` and `aarch64-linux-gnu-gcc`.
- **High-Core Graviton Tuning:**
  - Sizing `intra_op_threads` and `concurrent_runs` for 32/64-core Graviton systems.
  - NUMA node awareness: Pinning collector instances using `numactl --cpunodebind=0 --membind=0` to prevent inter-socket memory interconnect latency.
  - Setting `GOMEMLIMIT` and tuning `GOGC` for steady-state memory under high telemetry burst conditions.

---

### 4.4 GPU Server Integration (`docs/howto/gpu-acceleration.md`)

This guide addresses dual deployment architectures for GPU acceleration:

#### Pattern A: In-Process GPU Acceleration (Local ONNX Execution Providers)
- **NVIDIA CUDA & TensorRT:**
  - Dependencies: NVIDIA CUDA Toolkit 12.x, cuDNN 9.x, TensorRT 10.x.
  - Dynamic libraries: `libonnxruntime_providers_cuda.so`, `libonnxruntime_providers_tensorrt.so`.
  - CGo Shim Extension Architecture:
    - Detailed code blueprint showing how `ort_shim.c` registers the CUDA execution provider using `OrtSessionOptionsAppendExecutionProvider_CUDA(opts, device_id)`.
    - Memory arena configuration for GPU VRAM to avoid dynamic host-to-device memory allocation stalls.
  - Containerisation:
    - Base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
    - Docker run with NVIDIA Container Toolkit:
      ```bash
      docker run --gpus all --rm -p 4317:4317 -p 4318:4318 otelcol-onnxredactor:cuda
      ```
  - Kubernetes DaemonSet / Deployment:
    - Resource limits: `resources.limits: { nvidia.com/gpu: 1 }`.
    - Runtime class: `runtimeClassName: nvidia`.
- **AMD ROCm & MIGraphX:**
  - Dependencies: ROCm 6.x driver, MIOpen, MIGraphX.
  - Dynamic libraries: `libonnxruntime_providers_rocm.so`, `libonnxruntime_providers_migraphx.so`.
  - Containerisation with device mounts (`/dev/kfd`, `/dev/dri`).
  - Kubernetes device plugin: `resources.limits: { amd.com/gpu: 1 }`.
- **Failure Modes & Operational Mitigations:**
  - GPU Out of Memory (OOM) error isolation.
  - Fallback logic to CPU execution provider when GPU initialisation fails.
  - Thermal throttling, CUDA context resets, and recovery.

#### Pattern B: Distributed Remote Inference Offload (Triton Architecture)
- **Architectural Topology:**
  - High-fanout edge Collectors handle telemetry ingestion, static filtering, and WordPiece tokenisation.
  - Batched token tensors (`input_ids`, `attention_mask`) are dispatched over gRPC to a centralised NVIDIA Triton Inference Server cluster.
  - Edge Collectors receive output logits, compute argmax span classification, and apply in-place redaction masks.
- **Triton Model Repository Specification:**
  - Directory structure:
    ```text
    model_repository/
    └── ner_redactor/
        ├── 1/
        │   └── model.onnx
        └── config.pbtxt
    ```
  - Full `config.pbtxt` manifest including:
    - Dynamic batching (`max_queue_delay_microseconds: 2000`).
    - Max batch size (`max_batch_size: 128`).
    - Model input/output tensor declarations with variable sequence lengths (`[-1, -1]`).
    - GPU instance group allocation (`kind: KIND_GPU, count: 2`).
- **Comprehensive Trade-Off Analysis:**
  - Thorough evaluation comparing local SIMD CPU (~1.18 ms latency, 0 network overhead) against Remote Triton (~2.50 ms latency with network hop, but amortised throughput of 150k+ strings/sec across enterprise clusters).
  - Network overhead, serialization, MTU sizing, and gRPC keepalive tuning.

---

## 5. Build Asset & Automation Enhancements

To ensure the repository reflects the multi-platform documentation, the following build assets will be updated:

### 5.1 Multi-Architecture `Dockerfile`
Update the root `Dockerfile` to leverage Docker BuildKit multi-platform arguments:
- Replace hardcoded `FROM --platform=linux/arm64` with automatic target resolution:
  ```dockerfile
  FROM --platform=$BUILDPLATFORM golang:1.27-bookworm AS builder
  ARG TARGETARCH
  ARG TARGETOS
  ```
- Conditionally download either `onnxruntime-linux-x64-1.31.0.tgz` or `onnxruntime-linux-aarch64-1.31.0.tgz` based on `TARGETARCH`, verifying official SHA-256 checksums for both.
- Conditionally download `ocb` for `linux_amd64` or `linux_arm64` with verified checksums.
- Target runtime image: `FROM debian:bookworm-slim` preserving multi-arch support.

### 5.2 Enhanced `Makefile`
Update the repository `Makefile` to include:
- `download-ort`: Retains smart host detection (`Darwin/arm64`, `Darwin/x86_64`, `Linux/aarch64`, `Linux/x86_64`).
- Explicit targets for cross-architecture library acquisition:
  - `download-ort-linux-amd64`
  - `download-ort-linux-arm64`
  - `download-ort-darwin-arm64`
  - `download-ort-darwin-amd64`
- `print-env`: Helper target printing the exact `export CGO_CFLAGS=...`, `export CGO_LDFLAGS=...`, and library path commands for the current terminal session.

---

## 6. Language & Terminology Standards

All documents strictly adhere to Australian English spelling and grammatical conventions:
- Verbs ending in `-ise` rather than `-ize` (e.g. `optimise`, `quantise`, `tokenise`, `serialise`, `initialise`).
- Nouns ending in `-ence` / `-our` (e.g. `licence` (noun), `behaviour`).
- Technical artefacts spelled `artefacts`.
- Centred spelled `centred`.

---

## 7. Verification & Quality Gates

The documentation and build assets will be verified through the following checks:
1. **Link Integrity:** All relative markdown links between `README.md`, `docs/howto.md`, and `docs/howto/*.md` resolve without broken anchors.
2. **Spelling & Style Validation:** Australian English compliance across all markdown files.
3. **Build Script & Makefile Execution:** Verify `make download-ort` and `make print-env` execute cleanly without errors.
4. **Test Suite Integrity:** Unit tests pass with race detector enabled (`make test`).
5. **Multi-Arch Dockerfile Syntax Validation:** Validate that the revised Dockerfile parses with `docker buildx build --check .` or build syntax check.
