# Linux ARM64 Production Deployment Guide

This guide covers operational tuning, containerisation, and deployment strategies for running the OpenTelemetry Collector with the in-process `onnxredactionprocessor` on 64-bit ARM (`aarch64` / `arm64`) enterprise Linux servers.

All conventions and terminology in this document follow Australian English.

---

## 1. Supported Environments & Hardware Overview

### Target Cloud & Edge Platforms
- **Amazon Web Services (AWS):** Graviton 2 (`c6g`, `m6g`), Graviton 3 (`c7g`, `m7g`), and Graviton 4 (`c8g`, `m8g`) EC2 instances and EKS worker nodes.
- **Google Cloud Platform (GCP):** Tau T2A ARM instances.
- **Oracle Cloud Infrastructure (OCI):** Ampere Altra A1 compute shapes.
- **Microsoft Azure:** Cobalt 100 and Ampere Altra (Dpsv5, Epsv5) virtual machines.
- **Edge ARM64 Hardware:** Ampere eMAG/Altra bare-metal appliances, Raspberry Pi 5, and NVIDIA Jetson (running in CPU-only mode).

### Microarchitecture & SIMD Instructions

In-process transformer token classification is compute-bound. Modern 64-bit ARM architectures feature advanced vector extensions that dramatically accelerate INT8 neural inference:

1. **ARM NEON SIMD:** 128-bit vector registers available on all ARMv8-A processors.
2. **Dot Product (`asimddp` / `dotprod`):** Introduced in ARMv8.2-A (standard in Graviton 2+ and Ampere Altra). Calculates four 8-bit integer multiplications and accumulates into a 32-bit integer in a single clock cycle.
3. **Int8 Matrix Multiplication (`i8mm`):** Introduced in ARMv8.6-A (featured in AWS Graviton 3 and Graviton 4). Doubles INT8 matrix multiplication throughput compared to standard NEON.

Verify CPU features on your Linux ARM64 host:
```bash
grep -m1 -E 'asimddp|i8mm|fp16' /proc/cpuinfo
```

---

## 2. Multi-Architecture Container Build Pipelines

Because continuous delivery build runners often run on x86_64 architectures, use Docker Buildx to cross-compile production ARM64 images.

### Step 1: Initialise a Buildx Builder Instance
```bash
docker buildx create --name multiarch-builder --use
docker buildx inspect --bootstrap
```

### Step 2: Build and Push Multi-Architecture Image
Build natively for `linux/arm64` (or simultaneously for both `linux/amd64` and `linux/arm64`):
```bash
docker buildx build \
  --platform linux/arm64 \
  -t myregistry/otelcol-onnxredactor:latest \
  --load .
```

The multi-stage `Dockerfile` dynamically downloads the official `onnxruntime-linux-aarch64-1.31.0.tgz` archive and compiles the custom collector binary with native CGo bindings targeting `linux/arm64`.

---

## 3. Kubernetes Deployment on ARM64 Nodes

Deploy the collector using a Kubernetes Deployment with node affinity targeting `arm64` worker pools.

### Kubernetes Manifest (`otelcol-arm64.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: otelcol-onnxredactor
  namespace: monitoring
  labels:
    app: otelcol-onnxredactor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: otelcol-onnxredactor
  template:
    metadata:
      labels:
        app: otelcol-onnxredactor
    spec:
      nodeSelector:
        kubernetes.io/arch: arm64
      containers:
        - name: otelcol
          image: myregistry/otelcol-onnxredactor:latest
          imagePullPolicy: IfNotPresent
          env:
            - name: GOMEMLIMIT
              value: "140MiB"
          resources:
            requests:
              cpu: "1000m"
              memory: "160MiB"
            limits:
              cpu: "2000m"
              memory: "220MiB"
          ports:
            - name: otlp-grpc
              containerPort: 4317
            - name: otlp-http
              containerPort: 4318
            - name: metrics
              containerPort: 8888
          readinessProbe:
            httpGet:
              path: /metrics
              port: 8888
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /metrics
              port: 8888
            initialDelaySeconds: 15
            periodSeconds: 20
```

Deploy to the cluster:
```bash
kubectl apply -f otelcol-arm64.yaml
```

---

## 4. Performance Tuning & High Core-Count Graviton Optimisation

High-throughput aggregation nodes on AWS Graviton (e.g. `c7g.8xlarge` with 32 cores, or `c7g.16xlarge` with 64 cores) require specific tuning to prevent thread contention and cross-socket latency.

### 1. Sizing `intra_op_threads` and Worker Semaphores
Avoid setting `intra_op_threads` to the total core count of large multi-tenant nodes. High thread counts increase synchronisation overhead when evaluating small sequence lengths (32 to 256 tokens).

Recommended configuration in `config.yaml`:
```yaml
processors:
  onnxredaction:
    model:
      session:
        intra_op_threads: 4     # Pin intra-op execution to 4 physical cores
        concurrent_runs: 4      # Allow 4 concurrent inference passes
        allow_spinning: false   # Disables thread spinning when idle
    inference:
      coalesce_window: 2ms
      coalesce_max_strings: 64
      max_tensor_batch_size: 32
      length_buckets: [32, 128, 256]
```

### 2. NUMA Node Binding with `numactl`
On bare-metal multi-socket ARM servers (such as dual-socket Ampere Altra Max systems with 256 cores), memory access across NUMA (Non-Uniform Memory Access) nodes can add up to 40% latency overhead.

Pin the collector process to a single NUMA socket:
```bash
numactl --cpunodebind=0 --membind=0 /usr/local/bin/otelcol --config=/etc/otel/config.yaml
```

In `systemd`, configure NUMA affinity directly inside the unit file:
```ini
[Service]
NUMAPolicy=bind
NUMAMask=0
```

### 3. Tuning Go Memory Limits (`GOMEMLIMIT`)
In containerised Kubernetes environments with strict cgroup limits (e.g. 200 MiB limit):
- Set `GOMEMLIMIT=140MiB` (70% of the cgroup limit).
- The remaining 30% (~60 MiB) is preserved for native CGo ONNX Runtime memory arenas and dynamic library allocations, preventing out-of-memory (OOM) kills.
