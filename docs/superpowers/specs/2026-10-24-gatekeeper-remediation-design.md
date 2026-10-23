# Pre-Production Gatekeeper Review Remediation Specification

**Date:** 2026-10-24  
**Status:** Approved  
**Author:** Principal Systems Architect  
**Language Standard:** Australian English (documentation) / Go 1.27.2 (codebase)

---

## 1. Executive Summary

Following the 6-pillar pre-production gatekeeper review and `receiving-code-review` validation, this specification defines the exact technical remediations required to promote `vector-onnx-redactor` to production readiness.

---

## 2. Validated Scope & Architectural Decisions

### 2.1 P0 Critical Blockers

1. **P0-1: Canonical BIO Class Taxonomy Alignment (`processor.go`)**
   - `defaultIDToBIO` in `processor.go` must match the ONNX model specification (`docs/models/model-decision-record.md` and `scripts/generate_test_onnx.py`): index `0` is `"O"`, indices `1..9` are `B-PERSON` through `B-PATH_USER`, and indices `10..18` are `I-PERSON` through `I-PATH_USER`.

2. **P0-2: Runtime `Config.Entities` Allowlist Enforcement (`processor.go` & `coalesce.go`)**
   - Build a `[10]bool` allowed-label lookup table from `cfg.Entities`.
   - Apply the allowlist both when constructing `decode.NewDecoder` (mapping disabled classes to `"O"`) and inside `coalesce.Coalescer` (via `coalesce.WithAllowedLabels([10]bool)`) so both native ONNX engines and injected test detectors enforce `cfg.Entities`.

3. **P0-3: CGo Thread-Safe Initialisation & Output Tensor Shape Bounds (`ort_shim.c` & `engine.go`)**
   - Synchronise `ort_shim_init` using a package-level `sync.Once` in `internal/ort/engine.go`.
   - In `ort_shim_run_batch` (`ort_shim.c`), verify that the output tensor rank is 3 (`num_dims == 3`) and that `dims[0] == batch_size`, `dims[1] == seq_len`, and `dims[2] > 0` before computing `total_elements` and executing `memcpy`.

4. **P0-4: CGo Session Lifecycle Synchronisation (`engine.go`)**
   - Replace `e.mu sync.Mutex` with `e.mu sync.RWMutex` in `ort.Engine`.
   - `DetectBatch` acquires `e.mu.RLock()` before checking `e.session == nil` and holds `RLock()` across `C.ort_shim_run_batch` and the `runOptsPool` return `defer` (calling `e.mu.RUnlock()` without acquiring a write lock inside the `defer`), preventing concurrent `Close()` from freeing an active `OrtSession` while avoiding `RWMutex` self-deadlock.

5. **P0-5: Signal Coverage, Depth Bounding, Arena Sizing & Metric Collision Integrity**
   - **Trace Status Messages (`P0-5a`):** Add `TargetSpanStatusMessage` to `dedup.TargetKind` (`t.Span.Status().SetMessage(s)`) and collect `span.Status().Message()` in `processTraces`.
   - **Metric Exemplars (`P0-5b`):** Inspect `dp.Exemplars().At(e).FilteredAttributes()` across `Gauge`, `Sum`, `Histogram`, and `ExponentialHistogram` datapoints in `collectDataPointCandidates`.
   - **Nesting Depth Enforcement (`P0-5c`):** Preserve the intentional `ValueTypeBytes` skip behaviour, but when attribute nesting exceeds `maxNestingDepth` (32), record `redaction_skipped_total{reason="max_depth"}` and under `on_error: drop` scrub nested string values with `[REDACTED_UNINSPECTED]`.
   - **Metric Collision Resolution (`P0-5d`):**
     - Only invoke `metricsmerge.ResolveCollisions(m)` on a `pmetric.Metric` when at least one attribute in that metric was actually modified by redaction (preserving untouched multi-timestamp Gauge batches while retaining Last-Timestamp-Wins merging for collided redacted Gauge series).
     - In `ResolveCollisions` for `MetricTypeExponentialHistogram`, merge `ZeroCount()` and bucket counts (`Positive()` and `Negative()`) when scale and offset align.
   - **Dynamic Arena Sequence Length (`P0-5e`):** Initialise `arena.NewPool(cfg.Inference.MaxTensorBatchSize, cfg.Inference.WindowTokens, len(defaultIDToBIO))` in `newProcessor` rather than hardcoding `256`.

---

### 2.2 P1 & P2 Performance, Observability & Test Quality Enhancements

1. **P1-1 (Retained by Design):** Non-blocking semaphore acquisition in `coalesce.executeBatch` is intentionally preserved to satisfy fail-fast backpressure (`TestCoalescer_SemaphoreFull`).
2. **P1-2: End-to-End Golden Corpus Validation (`corpus_test.go`):**
   - Route all positive and negative samples in `TestCorpus_GoldenGatedPrecisionRecall` through `proc.ConsumeLogs`, recording detected spans inside `corpusDetector.DetectBatch` and asserting that mutated log bodies contain `[REDACTED_<LABEL>]` for positives and remain untouched for negatives.
3. **P1-3: Event-Driven Caller Cancellation Monitor (`coalesce.go`):**
   - Replace the 2 ms `time.Ticker` polling goroutine in `executeBatch` with a sequential channel wait over `req.ctx.Done()` across `activeRequests`.
   - Only spawn the CGo cancellation monitor goroutine in `ort.Engine.DetectBatch` when `ctx.Done() != nil`.
4. **P1-4: Hot-Path Allocation Reduction (`tokenizer.go` & `engine.go`):**
   - Use a stack-allocated `[256]byte` buffer in `wordPieceTokenizer.Encode` and `tokenizeWord` for ASCII lowercasing and `"##"` prefixing, enabling zero-allocation `map[string]int64` lookups.
   - Slice `a.SpanScratch` in `ort.Engine.DetectBatch` when calling `DecodeWindow` and eliminate the per-call `arena.NewPool` allocation in `Warmup`.
5. **P1-5: Structured Panic Logging (`coalesce.go`):**
   - Add `WithLogger(*zap.Logger) Option` to `coalesce.Coalescer` and log recovered panics with `zap.Any("panic", r)` and `zap.Stack("stacktrace")`.
6. **P2-1: Telemetry Lifecycle & Scrub Signal Attribution (`telemetry.go`, `dedup.go`, `coalesce.go`):**
   - Store `metric.Registration` on `telemetry` and unregister it in `telemetry.Close()` called from `redactionProcessor.Shutdown()`.
   - Store `Signal string` on `dedup.MutationTarget` so `RecordScrubbed` reports the exact signal (`"logs"`, `"traces"`, `"metrics"`) on `tokenizer_error`, `mask_error`, and `oversize_tail`.
7. **P2-2 & P2-3: Build & Dead Code Cleanup (`Makefile`, `arena/`, `types/`):**
   - Remove redundant `-lonnxruntime` / `-Wl,-rpath` flags from `Makefile` when `ORT_DIR` uses the default `.ort` path already specified in `engine.go` `#cgo` directives.
   - Remove unused `race_disabled.go` / `race_enabled.go` in `internal/arena` and unused duplicate target types in `internal/types/types.go`.

---

### 2.3 Documentation Audit Fixes (`README.md`, `docs/howto.md`, `docs/howto/macos.md`)

1. Update Go version prerequisites from `1.22+` to `1.27+` in `README.md`, `docs/howto.md`, and `docs/howto/macos.md`.
2. Update `docs/howto.md` to use `-Wl,-rpath,$(pwd)/.ort/lib` on macOS, `model_int8.onnx` in the export checksum command, and `/etc/otel/models/tiny_ner.onnx` in the sample configuration.
3. Clarify the local verification payload in `docs/howto/macos.md` so operators testing with the synthetic `tiny_ner.onnx` fixture use synthetic trigger tokens (`token_b-person`, `token_b-email`, `token_b-path_user`) or load an exported fine-tuned model for natural-language strings.
4. Ensure all markdown files pass `scripts/validate_australian_english.py`.
