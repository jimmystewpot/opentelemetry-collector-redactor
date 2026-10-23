# Platform & GPU Integration HOWTO Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a modular, production-grade HOWTO documentation suite covering macOS, Linux x86-64, Linux ARM64, and GPU acceleration (CUDA, ROCm, Triton) in Australian English, with build asset enhancements, automated language validation, and a pull request.

**Architecture:** Decompose the documentation into a dedicated `docs/howto/` suite led by an architectural decision matrix (`README.md`), platform-specific runbooks (`macos.md`, `linux-x86_64.md`, `linux-arm64.md`), and a comprehensive GPU integration guide (`gpu-acceleration.md`) addressing both in-process CGo providers and remote Triton offloading. Update `Makefile` and `Dockerfile` to support multi-arch builds natively, enforce Australian English via an automated validation script, and publish the work via a GitHub Pull Request.

**Tech Stack:** Markdown, Go 1.22+, CGo, OpenTelemetry Collector Builder (`ocb` v0.163.2), ONNX Runtime v1.31.0, Docker Buildx, Python 3, GitHub CLI (`gh`).

**Spec:** [`docs/superpowers/specs/2026-10-24-platform-and-gpu-howtos-design.md`](file:///Users/jlamb/go/src/github.com/jimmystewpot/vector-onnx-redactor/.worktrees/docs-howtos/docs/superpowers/specs/2026-10-24-platform-and-gpu-howtos-design.md)

## Global Constraints

- All documentation must adhere strictly to Australian English (en-AU) spelling conventions (`optimise`, `quantise`, `licence` (noun), `artefact`, `serialise`, `initialise`, `behaviour`, `colour`, `centre`).
- All relative markdown links between files must resolve correctly without 404 or broken fragment anchors.
- All Docker and shell snippets must be valid, runnable, and reference verified checksums for ONNX Runtime 1.31.0 and OCB 0.163.2.
- The repository test suite (`make test`) must pass cleanly at each step without regressions.
- Work must remain strictly isolated within the git worktree `docs/howtos-platform-gpu`.

## Review Focus

1. **Alpine Linux / musl dynamic linking failure:** Users attempting to run on Alpine Linux will encounter runtime crashes due to pre-built ONNX Runtime glibc requirements; documentation must explicitly flag this and provide Debian/Ubuntu alternatives.
2. **macOS System Integrity Protection (SIP) wiping `DYLD_LIBRARY_PATH`:** Spawning subprocesses or running under launchd strips `DYLD_LIBRARY_PATH`; documentation must specify `-Wl,-rpath` embedding and ad-hoc code signing (`codesign -s -`).
3. **GPU Out-of-Memory (OOM) during batch coalescing:** Large telemetry bursts on in-process GPU providers can exhaust VRAM; documentation must define memory arena limits and CPU fallback strategies.
4. **AWS Graviton / high core-count NUMA latency:** Inter-socket memory interconnect traversal on 64-core ARM systems degrades sub-millisecond p99 latency; documentation must detail `numactl` core and memory pinning.
5. **American English leakage in technical terms:** Terms like `optimize`, `quantize`, `license` (when used as a noun), `serialization` must be caught and prevented via automated linting.

---

### Task 1: Enhance Build Assets (Makefile & Dockerfile)

**Files:**
- Modify: `Dockerfile`
- Modify: `Makefile`

**Interfaces:**
- Consumes: Existing `Makefile` and `Dockerfile`.
- Produces: Multi-platform build capabilities (`linux/amd64`, `linux/arm64`), explicit download targets (`download-ort-linux-amd64`, `download-ort-linux-arm64`, `download-ort-darwin-arm64`, `download-ort-darwin-amd64`), and `print-env` helper.

- [ ] **Step 1: Update `Makefile` with platform-specific download targets and `print-env`**
  Add explicit targets for downloading ONNX Runtime archives per architecture and an environment printout target (`print-env`) that outputs exact `CGO_CFLAGS`, `CGO_LDFLAGS`, `LD_LIBRARY_PATH`, and `DYLD_LIBRARY_PATH`.

- [ ] **Step 2: Verify `Makefile` targets**
  Run: `make print-env`
  Expected: Outputs `export CGO_CFLAGS=...` and `export CGO_LDFLAGS=...`.

- [ ] **Step 3: Update `Dockerfile` for multi-platform BuildKit support**
  Refactor `Dockerfile` to accept `TARGETARCH` and `TARGETOS`, dynamically downloading the matching ONNX Runtime tarball and `ocb` binary with verified SHA-256 checksums (`amd64` and `arm64`).

- [ ] **Step 4: Verify Dockerfile syntax and existing test suite**
  Run: `docker build --check . && make test`
  Expected: Clean syntax check and passing Go tests.

- [ ] **Step 5: Commit changes**
  ```bash
  git add Makefile Dockerfile
  git commit -m "build: add multi-platform support to Makefile and Dockerfile"
  ```

---

### Task 2: Australian English Validation Script

**Files:**
- Create: `scripts/validate_australian_english.py`
- Test: `scripts/test_validate_australian_english.py`

**Interfaces:**
- Consumes: Markdown documentation files (`docs/**/*.md`).
- Produces: Python CLI tool exiting with 0 if all documents use Australian English, or 1 with line-numbered violations if American spellings are detected.

- [ ] **Step 1: Write test for Australian English validation script**
  Create `scripts/test_validate_australian_english.py` asserting that American words (`optimize`, `quantize`, `artifact`, `behavior`, `license` as noun, `serialize`, `center`) trigger errors, while Australian equivalents (`optimise`, `quantise`, `artefact`, `behaviour`, `licence`, `serialise`, `centre`) pass.

- [ ] **Step 2: Run test to verify it fails**
  Run: `python3 scripts/test_validate_australian_english.py`
  Expected: FAIL with missing module or function.

- [ ] **Step 3: Implement `scripts/validate_australian_english.py`**
  Implement regex scanner mapping American English spellings to Australian English equivalents, ignoring code blocks (fenced triple-backtick blocks) and inline code literals where exact library identifiers (e.g. `OrtArenaAllocator`, `cgo`) must remain intact.

- [ ] **Step 4: Run test to verify it passes**
  Run: `python3 scripts/test_validate_australian_english.py`
  Expected: PASS

- [ ] **Step 5: Commit changes**
  ```bash
  git add scripts/validate_australian_english.py scripts/test_validate_australian_english.py
  git commit -m "chore: add Australian English validation script and tests"
  ```

---

### Task 3: Author Master Platform Landing & Decision Matrix (`docs/howto/README.md`)

**Files:**
- Create: `docs/howto/README.md`

**Interfaces:**
- Consumes: Design spec Section 3.
- Produces: Main documentation landing page with platform decision tree, comparison matrix, latency/memory benchmarks, and navigation links.

- [ ] **Step 1: Write `docs/howto/README.md`**
  Author the master index document in Australian English covering:
  - Architecture overview and design philosophy.
  - Platform comparison matrix (macOS M-series vs Intel, Linux x86_64, Linux ARM64, In-process GPU, Remote Triton).
  - Decision tree based on throughput thresholds (< 10k, 10k–50k, > 50k strings/sec).
  - Navigation links to sub-guides (`macos.md`, `linux-x86_64.md`, `linux-arm64.md`, `gpu-acceleration.md`).

- [ ] **Step 2: Validate Australian English and verify links**
  Run: `python3 scripts/validate_australian_english.py docs/howto/README.md`
  Expected: PASS (0 violations).

- [ ] **Step 3: Commit changes**
  ```bash
  git add docs/howto/README.md
  git commit -m "docs: add master platform decision matrix and landing guide"
  ```

---

### Task 4: Author macOS Guide (`docs/howto/macos.md`)

**Files:**
- Create: `docs/howto/macos.md`

**Interfaces:**
- Consumes: Design spec Section 4.1.
- Produces: Complete macOS deployment and development guide for Apple Silicon (`arm64`) and Intel (`x86_64`).

- [ ] **Step 1: Write `docs/howto/macos.md`**
  Author the macOS guide in Australian English covering:
  - Prerequisites (`xcode-select`, Go 1.22+, `ocb`, Make).
  - Mach-O dynamic linking, `@rpath`, SIP clearance of `DYLD_LIBRARY_PATH`, and ad-hoc code-signing (`codesign -s -`).
  - Hardware acceleration on Apple Silicon (ARM NEON vs CoreML execution provider feasibility).
  - Step-by-step build recipe with `ocb --config=builder-config.yaml`.
  - Local verification with telemetry load generation and RSS memory inspection.

- [ ] **Step 2: Validate Australian English and test locally**
  Run: `python3 scripts/validate_australian_english.py docs/howto/macos.md && make test`
  Expected: PASS (0 violations, test suite passing on macOS).

- [ ] **Step 3: Commit changes**
  ```bash
  git add docs/howto/macos.md
  git commit -m "docs: add macOS development and deployment guide"
  ```

---

### Task 5: Author Linux x86_64 Guide (`docs/howto/linux-x86_64.md`)

**Files:**
- Create: `docs/howto/linux-x86_64.md`

**Interfaces:**
- Consumes: Design spec Section 4.2.
- Produces: Enterprise Linux x86_64 production deployment guide.

- [ ] **Step 1: Write `docs/howto/linux-x86_64.md`**
  Author the Linux x86_64 guide in Australian English covering:
  - Supported distributions (Debian 12, Ubuntu 22.04/24.04, RHEL/Rocky 9).
  - SIMD optimisations: AVX2, AVX-512, and VNNI instruction sets for INT8 quantised BERT models.
  - C ABI and linker nuances: Why GNU `glibc` is required and why Alpine `musl` causes runtime failures.
  - Production `systemd` unit file with unprivileged user `otel`, cgroups v2 limits (`MemoryHigh`, `MemoryMax`), and `GOMEMLIMIT`.
  - Production containerisation runbook.

- [ ] **Step 2: Validate Australian English**
  Run: `python3 scripts/validate_australian_english.py docs/howto/linux-x86_64.md`
  Expected: PASS (0 violations).

- [ ] **Step 3: Commit changes**
  ```bash
  git add docs/howto/linux-x86_64.md
  git commit -m "docs: add Linux x86_64 production deployment guide"
  ```

---

### Task 6: Author Linux ARM64 Guide (`docs/howto/linux-arm64.md`)

**Files:**
- Create: `docs/howto/linux-arm64.md`

**Interfaces:**
- Consumes: Design spec Section 4.3.
- Produces: Enterprise Linux ARM64 production deployment guide (AWS Graviton, Ampere Altra).

- [ ] **Step 1: Write `docs/howto/linux-arm64.md`**
  Author the Linux ARM64 guide in Australian English covering:
  - Target hardware: AWS Graviton 2/3/4, Ampere Altra, edge ARM nodes.
  - ARMv8.2-A+ Dot Product (`asimddp`) and ARMv8.6-A Int8 Matrix Multiplication (`i8mm`).
  - Multi-architecture container build recipes with `docker buildx`.
  - High core-count Graviton tuning: `intra_op_threads`, NUMA node affinity with `numactl`, and `GOMEMLIMIT` calibration.

- [ ] **Step 2: Validate Australian English**
  Run: `python3 scripts/validate_australian_english.py docs/howto/linux-arm64.md`
  Expected: PASS (0 violations).

- [ ] **Step 3: Commit changes**
  ```bash
  git add docs/howto/linux-arm64.md
  git commit -m "docs: add Linux ARM64 production deployment guide"
  ```

---

### Task 7: Author GPU Server Integration Guide (`docs/howto/gpu-acceleration.md`)

**Files:**
- Create: `docs/howto/gpu-acceleration.md`

**Interfaces:**
- Consumes: Design spec Section 4.4.
- Produces: Complete GPU acceleration and integration guide covering in-process execution providers and remote Triton offloading.

- [ ] **Step 1: Write `docs/howto/gpu-acceleration.md`**
  Author the GPU guide in Australian English covering:
  - **Pattern A (In-Process GPU):**
    - NVIDIA CUDA 12.x / TensorRT 10.x integration, cuDNN, dynamic libraries.
    - AMD ROCm 6.x / MIGraphX integration, `/dev/kfd` and `/dev/dri` device access.
    - CGo shim code blueprint for registering GPU execution providers in `ort_shim.c`.
    - Multi-stage GPU Dockerfile based on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
    - Kubernetes Pod specification with GPU resource limits (`nvidia.com/gpu: 1`).
  - **Pattern B (Remote Triton Offload):**
    - Distributed topology: Edge Collectors doing extraction/tokenisation -> gRPC -> Centralised Triton GPU cluster.
    - Full Triton Model Repository manifest (`config.pbtxt`) with dynamic batching (`max_queue_delay_microseconds: 2000`).
    - Comprehensive trade-off analysis: latency vs. throughput vs. infrastructure cost.
    - Resilience: VRAM OOM circuit breaking and fail-safe CPU fallbacks.

- [ ] **Step 2: Validate Australian English**
  Run: `python3 scripts/validate_australian_english.py docs/howto/gpu-acceleration.md`
  Expected: PASS (0 violations).

- [ ] **Step 3: Commit changes**
  ```bash
  git add docs/howto/gpu-acceleration.md
  git commit -m "docs: add GPU acceleration and server integration guide"
  ```

---

### Task 8: Update Root Guide (`docs/howto.md`) & Run Full Suite Verification

**Files:**
- Modify: `docs/howto.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: All completed docs under `docs/howto/`.
- Produces: Cohesive documentation tree with backwards-compatible root pointers and verified link integrity.

- [ ] **Step 1: Update `docs/howto.md` and `README.md`**
  Update `docs/howto.md` with an executive overview and direct navigation to the `docs/howto/` modular suite. Update root `README.md` to reference the new platform and GPU guides.

- [ ] **Step 2: Execute full Australian English validation across all markdown documentation**
  Run: `python3 scripts/validate_australian_english.py docs/ README.md`
  Expected: PASS across all files.

- [ ] **Step 3: Verify all tests and linting pass**
  Run: `make test && make lint`
  Expected: All tests pass with race detector enabled; linter passes.

- [ ] **Step 4: Commit changes**
  ```bash
  git add docs/howto.md README.md
  git commit -m "docs: update root documentation pointers to modular howto suite"
  ```

---

### Task 9: Create Pull Request

**Files:**
- Git branch: `docs/howtos-platform-gpu`

**Interfaces:**
- Consumes: All committed tasks on `docs/howtos-platform-gpu`.
- Produces: Open GitHub Pull Request into `main` with summary, verification evidence, and Australian English confirmation.

- [ ] **Step 1: Push worktree branch to remote origin**
  Run: `git push -u origin docs/howtos-platform-gpu`
  Expected: Branch pushed successfully.

- [ ] **Step 2: Create Pull Request with `gh pr create`**
  Run:
  ```bash
  gh pr create \
    --title "docs: add multi-platform and GPU integration HOWTO suite (Australian English)" \
    --body "..."
  ```
  Expected: Pull request URL returned.

- [ ] **Step 3: Verify PR status**
  Run: `gh pr view`
  Expected: PR open and ready for review.
