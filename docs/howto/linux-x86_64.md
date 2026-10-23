# Linux x86_64 Production Deployment Guide

This guide provides operational and configuration instructions for deploying the OpenTelemetry Collector with the in-process `onnxredactionprocessor` on enterprise 64-bit x86 (`x86_64` / `amd64`) Linux servers.

All conventions and terminology in this document follow Australian English.

---

## 1. Supported Distributions & System Requirements

### Target Operating Systems
- **Debian GNU/Linux:** Version 12 (Bookworm)
- **Ubuntu Server LTS:** Version 22.04 (Jammy Jellyfish), Version 24.04 (Noble Numbat)
- **Red Hat Enterprise Linux (RHEL) & Clones:** RHEL 9, Rocky Linux 9, AlmaLinux 9

### Hardware & CPU Prerequisites
- **Architecture:** 64-bit x86 (`x86_64`).
- **SIMD Vector Extensions:** Intel or AMD processors supporting **AVX2** (Advanced Vector Extensions 2) or **AVX-512**.
- **Deep Learning Boost (VNNI):** Processors with Vector Neural Network Instructions (`avx512_vnni` or `avx_vnni`) provide accelerated INT8 matrix dot-products, cutting inference latency by up to 35%.

Verify host processor flags using `grep`:
```bash
grep -m1 -E 'avx512_vnni|avx512f|avx2' /proc/cpuinfo
```

---

## 2. C Library ABI & Linker Nuances

The processor connects to Microsoft's ONNX Runtime shared library (`libonnxruntime.so`) via CGo. Operating native code on Linux requires careful attention to C library runtimes and dynamic linker search paths.

### GNU C Library (`glibc`) vs. Alpine (`musl`)

> [!WARNING]
> Official pre-compiled ONNX Runtime Linux releases (`onnxruntime-linux-x64-*.tgz`) are compiled against GNU `glibc`.
> 
> Running this binary inside Alpine Linux (`musl libc`) will result in dynamic linker failures (e.g. `Error relocating /usr/local/lib/libonnxruntime.so: __isnanf: symbol not found`) or segmentation faults.
> 
> Always base your production containers and virtual machine hosts on **Debian**, **Ubuntu**, or **RHEL** distributions utilising `glibc`.

### System-Wide Dynamic Linker Configuration

To avoid relying on volatile per-session `LD_LIBRARY_PATH` variables in production environments, register the ONNX Runtime library directory in the system linker cache:

1. Unpack the ONNX Runtime library into `/opt/onnxruntime`:
   ```bash
   sudo mkdir -p /opt/onnxruntime
   sudo tar -xzf onnxruntime-linux-x64-1.31.0.tgz -C /opt/onnxruntime --strip-components=1
   ```

2. Create an `ld.so` configuration file:
   ```bash
   echo "/opt/onnxruntime/lib" | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
   ```

3. Update the runtime linker cache:
   ```bash
   sudo ldconfig
   ```

4. Verify that `libonnxruntime.so` is resolved by the system dynamic linker:
   ```bash
   ldconfig -p | grep libonnxruntime
   ```

---

## 3. Host Deployment via systemd

For bare-metal servers or enterprise virtual machines running without Kubernetes, deploy the collector as a hardened `systemd` service with native cgroups v2 resource boundaries.

### Step 1: Install Collector Binary & Configuration

1. Place the compiled binary in `/usr/local/bin`:
   ```bash
   sudo cp bin/otelcol-onnxredactor /usr/local/bin/otelcol
   sudo chmod 755 /usr/local/bin/otelcol
   ```

2. Create system user, group, and required directories:
   ```bash
   sudo groupadd --system otel
   sudo useradd --system -g otel -d /var/lib/otel -s /bin/false otel
   sudo mkdir -p /etc/otel/models /var/log/otel
   ```

3. Copy configuration file and neural model artefacts:
   ```bash
   sudo cp config.yaml /etc/otel/config.yaml
   sudo cp processor/onnxredactionprocessor/testdata/tiny_ner.onnx /etc/otel/models/tiny_ner.onnx
   sudo cp processor/onnxredactionprocessor/testdata/tokenizer.json /etc/otel/models/tokenizer.json
   sudo chown -R otel:otel /etc/otel /var/log/otel
   ```

### Step 2: Configure the systemd Service Unit

Create `/etc/systemd/system/otelcol-onnxredactor.service`:

```ini
[Unit]
Description=OpenTelemetry Collector with In-Process ONNX Redactor
Documentation=https://github.com/jimmystewpot/vector-onnx-redactor
After=network.target

[Service]
Type=simple
User=otel
Group=otel
WorkingDirectory=/etc/otel
ExecStart=/usr/local/bin/otelcol --config=/etc/otel/config.yaml
Restart=on-failure
RestartSec=5s

# Memory Management & Limits (cgroups v2)
# GOMEMLIMIT reserves headroom for native CGo ONNX Runtime memory arenas
Environment=GOMEMLIMIT=140MiB
MemoryAccounting=true
MemoryHigh=180M
MemoryMax=220M

# CPU Accounting & Concurrency
CPUAccounting=true
LimitNOFILE=65536

# Security Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
RestrictNamespaces=true
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE
ReadWritePaths=/var/log/otel

[Install]
WantedBy=multi-user.target
```

### Step 3: Enable and Start the Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now otelcol-onnxredactor.service
```

Inspect service health and runtime logs:
```bash
sudo systemctl status otelcol-onnxredactor.service
sudo journalctl -u otelcol-onnxredactor.service -f
```

---

## 4. Containerised Production Deployment (Docker / OCI)

### Building the Image
Compile the container using the multi-platform BuildKit Dockerfile:
```bash
docker build -t otelcol-onnxredactor:latest .
```

### Running the Container with Resource Ceilings
Launch the collector container with enforced memory and CPU limits:
```bash
docker run -d \
  --name otelcol-redactor \
  --restart unless-stopped \
  --memory="200m" \
  --memory-reservation="160m" \
  --cpus="2" \
  -e GOMEMLIMIT=140MiB \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 8888:8888 \
  otelcol-onnxredactor:latest
```

### Inspecting Container Runtime Metrics
Check memory consumption and cgroup limits:
```bash
docker stats --no-stream otelcol-redactor
```

---

## 5. Performance Tuning & CPU Sizing

1. **`intra_op_threads`:**
   Configure `intra_op_threads` in `config.yaml` to match the exact number of physical CPU cores allocated to the collector process. For example, on a 4-core instance, set:
   ```yaml
   session:
     intra_op_threads: 4
     concurrent_runs: 2
     allow_spinning: false
   ```

2. **Disable Thread Spinning (`allow_spinning: false`):**
   Ensures ONNX Runtime worker threads do not busy-wait when the batch coalescer queue is idle, conserving host CPU power for other workloads.

3. **Coalescing Calibration:**
   On multi-threaded x86_64 aggregation gateways handling high volume, calibrate the coalescing window to maximise SIMD vector fill rates:
   ```yaml
   inference:
     coalesce_window: 3ms
     coalesce_max_strings: 128
     max_tensor_batch_size: 32
     length_buckets: [32, 128, 256]
   ```
