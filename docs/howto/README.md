# Platform Deployment & Hardware Acceleration Guides

This suite of guides provides operational, architectural, and deployment instructions for running the OpenTelemetry Collector with the in-process `onnxredactionprocessor`.

All documentation in this suite strictly adheres to Australian English spelling conventions.

---

## 1. Architectural Overview & Design Philosophy

The `vector-onnx-redactor` processor executes neural Named Entity Recognition (NER) directly inside the OpenTelemetry Collector daemon via a zero-allocation CGo bridge to Microsoft's ONNX Runtime C API.

Because inference runs in-process or across tightly coupled native interfaces, compute throughput and latency depend directly on the underlying CPU SIMD instruction sets (ARM NEON, Intel/AMD AVX2 and AVX-512 VNNI) or specialised GPU accelerators (NVIDIA CUDA/TensorRT and AMD ROCm/MIGraphX).

Selecting the appropriate deployment topology requires balancing ingestion volume, end-to-end telemetry latency budgets, container memory ceilings, and operational infrastructure complexity.

---

## 2. Platform Comparison Matrix

| Platform / Topology | Compute Backend | Best Suited For | Typical p99 Latency (Batch of 32) | Memory Overhead | Operational Complexity | Guide |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **macOS (Apple Silicon)** | Apple M-series CPU (NEON) | Local engineering workstations, CI runners, edge servers | ~1.18 ms | < 200 MB RSS | Low | [macOS Guide](macos.md) |
| **macOS (Intel x86_64)** | Intel Core / Xeon (AVX2) | Legacy workstation development and validation | ~1.85 ms | < 200 MB RSS | Low | [macOS Guide](macos.md) |
| **Linux x86_64** | Intel Xeon / AMD EPYC (AVX-512 VNNI) | Central enterprise aggregation gateways, on-prem Kubernetes | ~1.05 ms | < 250 MB RSS | Medium | [Linux x86_64 Guide](linux-x86_64.md) |
| **Linux ARM64** | AWS Graviton 3/4, Ampere Altra (NEON) | Cloud-native edge collectors, cost-optimised Kubernetes clusters | ~1.12 ms | < 200 MB RSS | Medium | [Linux ARM64 Guide](linux-arm64.md) |
| **In-Process GPU (CUDA)** | NVIDIA L4 / T4 / A10G | Dedicated high-volume gateway nodes (> 50k strings/s) | ~0.45 ms | > 1.5 GB VRAM | High | [GPU Integration](gpu-acceleration.md) |
| **In-Process GPU (ROCm)** | AMD Instinct MI210 / MI300 | Open-ecosystem high-volume telemetry clusters | ~0.50 ms | > 2.0 GB VRAM | High | [GPU Integration](gpu-acceleration.md) |
| **Remote Triton Offload** | Dedicated GPU Server Cluster | Multi-tenant fleets, ultra-lightweight edge collectors (< 80MB) | ~2.50 ms (incl. RTT) | Edge: < 80 MB | High | [GPU Integration](gpu-acceleration.md) |

---

## 3. Architectural Decision Framework

```
                          [ Telemetry Ingestion Rate per Collector Node ]
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             Under 50,000 strings/sec                          Over 50,000 strings/sec
                        │                                                 │
            In-Process CPU Execution                            GPU Hardware Acceleration
                        │                                                 │
        ┌───────────────┴───────────────┐                 ┌───────────────┴───────────────┐
        ▼                               ▼                 ▼                               ▼
 Cloud / Edge ARM64             Enterprise x86_64    Dedicated Gateway Pods     Memory-Constrained Edge
 (AWS Graviton, Ampere)         (AVX2 / AVX-512)     (Single node > 50k str/s)   (Pods capped at < 100 MB)
        │                               │                 │                               │
        ▼                               ▼                 ▼                               ▼
 [ Linux ARM64 Guide ]        [ Linux x86_64 Guide ]  [ Pattern A: In-Process GPU ]   [ Pattern B: Remote Triton ]
```

### Decision Rules:

1. **Volume < 10,000 candidate strings/sec per node:**
   - Deploy **In-Process CPU** execution on standard cloud compute (e.g. AWS Graviton `c7g` or Intel/AMD `c6i`/`c6a`).
   - Achieves sub-1.2 ms p99 batch latency with zero GPU driver operational overhead.
   - Recommended coalescing settings: `coalesce_window: 2ms`, `coalesce_max_strings: 64`.

2. **Volume between 10,000 and 50,000 strings/sec per node:**
   - Scale CPU horizontally using Kubernetes Horizontal Pod Autoscaler (HPA) targeting 70% CPU utilisation.
   - Tune batch coalescing for higher batch fill rates: `coalesce_window: 4ms`, `coalesce_max_strings: 128`.
   - Allocate 4 to 8 CPU cores per collector pod with `intra_op_threads: 4`.

3. **Volume > 50,000 strings/sec per node or sub-500µs SLA:**
   - Deploy **Pattern A (In-Process GPU Acceleration)** on GPU-enabled instances (e.g. NVIDIA L4 on AWS `g6` or GKE).
   - Offloads tensor compute directly to GPU CUDA/TensorRT cores while keeping tokenisation and masking in-memory.

4. **Edge Collectors with strict memory limits (< 100 MB RAM):**
   - Deploy **Pattern B (Remote Triton Inference Offloading)**.
   - Edge daemons parse attributes, perform static filtering, and ship compact token arrays to a centralised Triton GPU pool.

---

## 4. Documentation Suite Navigation

Explore the dedicated deployment and operational guides:

- **[macOS Deployment Guide](macos.md):**
  Setting up Apple Silicon (`arm64`) and Intel (`x86_64`) environments, Xcode command line tools, Mach-O `@rpath` dynamic linking, System Integrity Protection (SIP) caveats, and local OpenTelemetry Collector Builder (`ocb`) compilation.

- **[Linux x86_64 Production Guide](linux-x86_64.md):**
  Targeting Debian, Ubuntu, and RHEL/Rocky Linux. Covers AVX2 and AVX-512 VNNI vector instruction optimisations, GNU `glibc` vs. Alpine `musl` ABI compatibility requirements, production `systemd` unit configuration with cgroups v2, and containerisation.

- **[Linux ARM64 Production Guide](linux-arm64.md):**
  Deploying on AWS Graviton 2/3/4 and Ampere Altra instances. Covers ARM NEON and I8MM matrix multiplication extensions, multi-architecture Docker Buildx pipelines, and NUMA node core pinning.

- **[GPU Acceleration & Server Integration Guide](gpu-acceleration.md):**
  Comprehensive coverage of GPU workloads:
  - *Pattern A (In-Process GPU):* NVIDIA CUDA 12.x / TensorRT and AMD ROCm 6.x execution providers, CGo bridge configuration, NVIDIA Container Toolkit, and Kubernetes DaemonSet manifests.
  - *Pattern B (Distributed Remote Triton Offload):* Architecting a centralised NVIDIA Triton Inference Server cluster, dynamic batching configuration (`config.pbtxt`), latency vs. network overhead trade-off analysis, and failover resilience.
