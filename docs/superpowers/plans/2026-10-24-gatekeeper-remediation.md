# Pre-Production Gatekeeper Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all validated P0 critical blockers, P1/P2 performance and observability findings, and documentation discrepancies identified in the pre-production gatekeeper review.

**Architecture:** We harden the CGo ONNX Runtime bridge with `sync.Once` initialisation, strict 3D output tensor bounds checking, and `sync.RWMutex` session lifecycle protection; eliminate hot-path heap allocations in WordPiece tokenisation and sub-batch cancellation monitoring; enforce `Config.Entities` allowlists and complete OTel signal coverage (trace status messages, metric exemplars, deep nesting bounds, conditional metric collision merging, and `ExponentialHistogram` bucket merging); and align all runbooks with Australian English and Go 1.27+.

**Tech Stack:** Go 1.27.2, CGo (`onnxruntime_c_api.h` v1.31.0), OpenTelemetry Collector SDK v1.69.0 / v0.163.0, Python 3.10+ (`validate_australian_english.py`), GitHub CLI (`gh`).

**Spec:** `docs/superpowers/specs/2026-10-24-gatekeeper-remediation-design.md`

## Global Constraints

- **Workspace Isolation:** Execute all tasks inside `/Users/jlamb/go/src/github.com/jimmystewpot/vector-onnx-redactor/.worktrees/gatekeeper-remediation` on branch `fix/gatekeeper-remediation`.
- **Go Version:** `go 1.27.2` (`processor/onnxredactionprocessor/go.mod`).
- **Australian English:** All markdown files under `docs/` and `README.md` must pass `python3 scripts/validate_australian_english.py docs/ README.md` (`make validate-spelling`).
- **Zero Race & Linter Regressions:** Every task must pass `make test` (`go test -race -count=1 ./...`) and `make lint` (`golangci-lint run ./...`) with 0 issues.
- **Preserved Backpressure Contract:** Non-blocking semaphore acquisition in `coalesce.executeBatch` (`TestCoalescer_SemaphoreFull`) and `ValueTypeBytes` log body skip behaviour (`TestProcessor_LogBodyTypes`) must remain intact.

## Review Focus

- **Concurrent `Engine.Close()` during active `Engine.DetectBatch()`:** Calling `Close()` while a slow batch is executing in CGo must wait for `ort_shim_run_batch` to return before freeing the `OrtSession`, without self-deadlocking when returning `runOpts` to `runOptsPool`.
- **Multi-timestamp unredacted Gauge metrics:** A `pmetric.Metric` of type `Gauge` containing multiple data points for the same series across distinct timestamps must not collapse into a single data point when zero attributes in that metric were modified by redaction.
- **Configured `entities` subset with both real ONNX engine and injected `Detector`:** Configuring `entities: ["EMAIL"]` must suppress `PERSON` and `PATH_USER` spans regardless of whether spans originate from `decode.Decoder` or a custom `detect.Detector`.
- **Deeply nested map/slice attributes exceeding `maxNestingDepth` (32):** Under `on_error: drop`, string values nested beyond depth 32 must be scrubbed in-place with `[REDACTED_UNINSPECTED]` and increment `redaction_skipped_total{reason="max_depth"}`.
- **Uppercase and subword tokens exceeding stack buffer capacity in `wordPieceTokenizer`:** Words up to `maxInputChar` (100 bytes) with `"##"` prefix must tokenise with 0 heap allocations, while words exceeding `maxInputChar` safely emit `unkID`.

---

### Task 1: CGo Native Thread-Safety, Shape Validation, RWMutex Lifecycle & Dead Code Cleanup

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/ort/ort_shim.c:11-41,260-298`
- Modify: `processor/onnxredactionprocessor/internal/ort/engine.go:30-199,225-296`
- Modify: `processor/onnxredactionprocessor/internal/ort/engine_test.go`
- Delete: `processor/onnxredactionprocessor/internal/arena/race_disabled.go`
- Delete: `processor/onnxredactionprocessor/internal/arena/race_enabled.go`
- Modify: `processor/onnxredactionprocessor/internal/types/types.go:115-152`
- Modify: `Makefile:34-38`

**Interfaces:**
- Consumes: `C.ort_shim_init(**C.char) C.int`, `arena.BatchArena.SpanScratch []types.EntitySpan`
- Produces: Thread-safe `ort.NewEngine`, `(*Engine).DetectBatch`, and `(*Engine).Close` using `sync.RWMutex` and `sync.Once`.

- [ ] **Step 1: Write failing tests in `processor/onnxredactionprocessor/internal/ort/engine_test.go` for concurrent `NewEngine` init, concurrent `DetectBatch` + `Close()`, and `SpanScratch` reuse**

```go
func TestEngine_ConcurrentCloseDuringInference(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(slowModelPath), slowSHA, 1, 2, false, dec)
	require.NoError(t, err)

	pool := arena.NewPool(2, 8, 19)
	a := pool.Acquire()
	defer pool.Release(a)

	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 102},
		Offsets:         []types.Offset{{Start: 0, End: 0}, {Start: 0, End: 4}, {Start: 4, End: 4}},
	}
	batch := types.BucketedBatch{MaxSeqLen: 3, Windows: []types.TokenWindow{w}}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = eng.DetectBatch(ctx, batch, a)
	}()

	time.Sleep(2 * time.Millisecond)
	require.NoError(t, eng.Close())
	<-done

	_, errAfter := eng.DetectBatch(context.Background(), batch, a)
	require.ErrorContains(t, errAfter, "session is closed")
}
```

- [ ] **Step 2: Run test to verify behaviour under `-race`**

Run: `cd processor/onnxredactionprocessor && go test -race -v -run TestEngine_ConcurrentCloseDuringInference ./internal/ort`
Expected: Race detector or unprotected concurrent access on `e.session` between `DetectBatch` and `Close`.

- [ ] **Step 3: Harden `ort_shim.c` and `engine.go`, remove dead code in `arena`/`types`, and deduplicate flags in `Makefile`**

1. In `processor/onnxredactionprocessor/internal/ort/ort_shim.c` (`ort_shim_run_batch`, lines 261–285):
   - Require `num_dims == 3 && dims[0] == batch_size && dims[1] == seq_len && dims[2] > 0`; otherwise release `shape_info` and `out_tensors[0]`, set `*out_err = strdup("ort_shim: unexpected output tensor dimensions")`, and return `-1`.
2. In `processor/onnxredactionprocessor/internal/ort/engine.go`:
   - Add package-level `var ( initOnce sync.Once; initErr error )` calling `C.ort_shim_init` once before `C.ort_shim_create_session`.
   - Change `Engine.mu` from `sync.Mutex` to `sync.RWMutex` and add `warmupArena *arena.BatchArena` initialised in `NewEngine` (`arena.NewPool(1, 2, 19).Acquire()`).
   - In `DetectBatch`: acquire `e.mu.RLock()`, check `if e.session == nil { e.mu.RUnlock(); return nil, errors.New("ort: session is closed") }`, and in the `defer` unset terminate, return `runOpts` via non-blocking `select { case e.runOptsPool <- runOpts: default: C.ort_shim_release_run_options(runOpts) }`, and call `e.mu.RUnlock()` (do **not** call `e.mu.Lock()` inside `RLock`).
   - Only spawn the cancellation monitor goroutine in `DetectBatch` when `ctx.Done() != nil`.
   - In `DetectBatch` lines 284–293: reset `a.SpanScratch = a.SpanScratch[:0]` and pass the remaining capacity slice `a.SpanScratch[startIdx:startIdx]` as `dst` to `e.decoder.DecodeWindow`, updating `a.SpanScratch` so decoded spans use the arena buffer.
3. Delete `processor/onnxredactionprocessor/internal/arena/race_disabled.go` and `processor/onnxredactionprocessor/internal/arena/race_enabled.go`.
4. Remove unused `SignalKind`, `TargetKind`, and `MutationTarget` from `processor/onnxredactionprocessor/internal/types/types.go` (lines 115–152).
5. In `Makefile` lines 34–38: remove `export CGO_CFLAGS` and `export CGO_LDFLAGS` when `ORT_DIR` equals `$(CURDIR)/.ort` (since `engine.go` `#cgo` directives already set `-I`, `-L`, `-lonnxruntime`, and `-Wl,-rpath` for `.ort`), eliminating duplicate linker warnings.

- [ ] **Step 4: Run tests and linter to verify Task 1**

Run: `make test && make lint`
Expected: PASS with 0 linker warnings and 0 linter issues.

- [ ] **Step 5: Commit Task 1**

```bash
git add Makefile processor/onnxredactionprocessor/internal/ort/ processor/onnxredactionprocessor/internal/arena/ processor/onnxredactionprocessor/internal/types/
git commit -m "fix(ort): synchronize CGo session lifecycle, validate 3D output shape, and remove dead code"
```

---

### Task 2: Zero-Allocation WordPiece Lowercasing & Subword Prefixing

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:125-242`
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`

**Interfaces:**
- Consumes: `types.Offset`
- Produces: Zero-allocation `(*wordPieceTokenizer).Encode(text string, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset, error)`

- [ ] **Step 1: Write failing allocation test and benchmark in `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`**

```go
func TestWordPieceTokenizer_ZeroAllocations(t *testing.T) {
	tok, err := NewWordPieceTokenizer(filepath.Join("..", "..", "testdata", "tokenizer.json"), "")
	require.NoError(t, err)

	ids := make([]int64, 0, 64)
	offsets := make([]types.Offset, 0, 64)
	input := "User AliceSmith and BobDoe logged in from HTTP API"

	allocs := testing.AllocsPerRun(100, func() {
		ids, offsets, err = tok.Encode(input, ids[:0], offsets[:0])
		if err != nil {
			t.Fatal(err)
		}
	})
	assert.Equal(t, float64(0), allocs, "WordPiece Encode must perform 0 heap allocations with preallocated buffers")
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestWordPieceTokenizer_ZeroAllocations ./internal/tokenizer`
Expected: FAIL (`allocs > 0` due to `strings.ToLower` and `t.prefix + subStr`).

- [ ] **Step 3: Implement stack-buffered lowercasing and subword prefixing in `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go`**

In `Encode` and `tokenizeWord`:
- Check word byte length `wordEnd - wordStart` (or lowercased length) against `t.maxInputChar` first.
- Use a stack buffer `var buf [256]byte` where `prefixLen := len(t.prefix)` is reserved at the start (`buf[prefixLen:]` holds the lowercased word bytes; ASCII `'A'..'Z'` is lowercased inline via `b + ('a' - 'A')`, falling back to `utf8.EncodeRune(buf[n:], unicode.ToLower(r))` for non-ASCII runes).
- For whole-word lookup, index `t.vocab[string(buf[prefixLen:prefixLen+wordLen])]`.
- For subword lookup when `subStart > 0`, copy `t.prefix` into `buf[prefixLen+subStart-prefixLen : prefixLen+subStart]`, look up `t.vocab[string(buf[subStart : prefixLen+subEnd])]`, and restore the overwritten bytes before advancing `subStart`.

- [ ] **Step 4: Run tokeniser tests and verify 0 allocations**

Run: `cd processor/onnxredactionprocessor && go test -race -v ./internal/tokenizer`
Expected: PASS (`TestWordPieceTokenizer_ZeroAllocations` passes with 0 allocs; note `-race` may instrument map accesses so gate `testing.AllocsPerRun` or run without `-race` to confirm 0 allocs, and ensure `make test` passes).

- [ ] **Step 5: Commit Task 2**

```bash
git add processor/onnxredactionprocessor/internal/tokenizer/
git commit -m "perf(tokenizer): eliminate heap allocations in WordPiece lowercasing and subword lookup"
```

---

### Task 3: Event-Driven Cancellation, Entity Allowlist, Structured Panic Logging & Scrub Telemetry in Coalescer

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/dedup/dedup.go:10-44`
- Modify: `processor/onnxredactionprocessor/internal/dedup/dedup_test.go`
- Modify: `processor/onnxredactionprocessor/internal/coalesce/coalesce.go:62-173,284-542`
- Modify: `processor/onnxredactionprocessor/internal/coalesce/coalesce_test.go`
- Modify: `processor/onnxredactionprocessor/telemetry.go:14-35,160-195`

**Interfaces:**
- Consumes: `types.Label`, `*zap.Logger`, `metric.Registration`
- Produces:
  - `dedup.TargetSpanStatusMessage` (`TargetKind`), `dedup.MutationTarget.Signal string`
  - `coalesce.WithAllowedLabels(allowed [10]bool) Option`
  - `coalesce.WithLogger(logger *zap.Logger) Option`
  - `(*telemetry).Close() error`

- [ ] **Step 1: Write failing tests in `dedup_test.go` and `coalesce_test.go`**

1. In `processor/onnxredactionprocessor/internal/dedup/dedup_test.go`, add a test case asserting `TargetSpanStatusMessage` mutates `span.Status().Message()`.
2. In `processor/onnxredactionprocessor/internal/coalesce/coalesce_test.go`, add `TestCoalescer_AllowedLabelsFiltering` and `TestCoalescer_ScrubbedSignalAttribution`:

```go
func TestCoalescer_AllowedLabelsFiltering(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()
	fakeDet.SpansToReturn = [][]types.EntitySpan{
		{
			{Start: 0, End: 5, Label: types.LabelPerson, Confidence: 0.99},
			{Start: 6, End: 23, Label: types.LabelEmail, Confidence: 0.99},
		},
	}

	var allowed [10]bool
	allowed[types.LabelEmail] = true // Only EMAIL enabled; PERSON disabled

	c := New(5*time.Millisecond, 1, nil, tok, chunker, bucketer, fakeDet, pool, 16384, types.ErrorModeDrop, false, WithAllowedLabels(allowed))
	defer func() { _ = c.Close() }()

	val := pcommon.NewValueStr("Alice alice@example.com")
	err := c.Submit(context.Background(), []Item{{Signal: "logs", Text: val.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val, Signal: "logs"}}})
	assert.NoError(t, err)
	assert.Equal(t, "Alice [REDACTED_EMAIL]", val.Str())
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestCoalescer_AllowedLabelsFiltering ./internal/coalesce`
Expected: FAIL (`WithAllowedLabels` undefined).

- [ ] **Step 3: Implement `TargetSpanStatusMessage`, `WithAllowedLabels`, `WithLogger`, event-driven cancellation, and `telemetry.Close()`**

1. In `processor/onnxredactionprocessor/internal/dedup/dedup.go`:
   - Add `TargetSpanStatusMessage` to `TargetKind` and handle `case TargetSpanStatusMessage: t.Span.Status().SetMessage(s)` in `(*MutationTarget).Set(s string)`.
   - Add `Signal string` field to `MutationTarget`.
2. In `processor/onnxredactionprocessor/internal/coalesce/coalesce.go`:
   - Add `allowedLabels [10]bool`, `hasAllowedLabels bool`, and `logger *zap.Logger` to `Coalescer`.
   - Implement `WithAllowedLabels(allowed [10]bool) Option` and `WithLogger(logger *zap.Logger) Option`.
   - In `executeBatch` panic `defer`: if `c.logger != nil`, call `c.logger.Error("redaction batch panic recovered", zap.Any("panic", r), zap.Stack("stacktrace"))`.
   - Replace the 2 ms `time.NewTicker` goroutine (lines 450–467) with a sequential wait over `activeRequests`:
     ```go
     go func() {
         defer cancel()
         for _, req := range activeRequests {
             select {
             case <-stopMonitor:
                 return
             case <-ctx.Done():
                 return
             case <-req.ctx.Done():
             }
         }
     }()
     ```
   - When appended to `parentSpans[pIdx]` (lines 487–492), if `c.hasAllowedLabels`, filter `spansSlice[wIdx]` so only spans with `s.Label < types.Label(len(c.allowedLabels)) && c.allowedLabels[s.Label]` are kept.
   - For `tokenizer_error`, `mask_error`, and `oversize_tail`, call `c.telemetry.RecordScrubbed(trg.Signal, reason)` inside the `for _, trg := range a.Dedup.TargetsFor(idx)` loop.
3. In `processor/onnxredactionprocessor/telemetry.go`:
   - Store `reg metric.Registration` from `meter.RegisterCallback` on `telemetry` and add `func (t *telemetry) Close() error` calling `t.reg.Unregister()`.

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run: `make test && make lint`
Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add processor/onnxredactionprocessor/internal/dedup/ processor/onnxredactionprocessor/internal/coalesce/ processor/onnxredactionprocessor/telemetry.go
git commit -m "feat(coalesce): add entity allowlist filtering, event-driven cancellation, panic logging, and signal scrub metrics"
```

---

### Task 4: Metric Collision Resolution — ExponentialHistogram Bucket & ZeroCount Merging

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge.go:217-255`
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go:158-190`

**Interfaces:**
- Consumes: `pmetric.ExponentialHistogramDataPoint`, `pmetric.ExponentialHistogramDataPointBuckets`
- Produces: Complete `ResolveCollisions` for `MetricTypeExponentialHistogram` merging `ZeroCount`, `Positive`, and `Negative` buckets.

- [ ] **Step 1: Update `TestResolveCollisions_ExponentialHistogram` in `processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go` to assert `ZeroCount` and bucket merging**

```go
	dp1.SetZeroCount(2)
	dp1.Positive().SetOffset(1)
	dp1.Positive().BucketCounts().FromRaw([]uint64{3, 5})
	dp1.Negative().SetOffset(0)
	dp1.Negative().BucketCounts().FromRaw([]uint64{1})

	dp2.SetZeroCount(1)
	dp2.Positive().SetOffset(1)
	dp2.Positive().BucketCounts().FromRaw([]uint64{2, 2})
	dp2.Negative().SetOffset(0)
	dp2.Negative().BucketCounts().FromRaw([]uint64{4})
```
And assert after `ResolveCollisions(m)`:
```go
	assert.Equal(t, uint64(3), dpMerged.ZeroCount())
	assert.Equal(t, []uint64{5, 7}, dpMerged.Positive().BucketCounts().AsRaw())
	assert.Equal(t, []uint64{5}, dpMerged.Negative().BucketCounts().AsRaw())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestResolveCollisions_ExponentialHistogram ./internal/metricsmerge`
Expected: FAIL (`ZeroCount` is `2` instead of `3`, `Positive().BucketCounts()` is `[3, 5]` instead of `[5, 7]`).

- [ ] **Step 3: Implement `ZeroCount` and bucket merging in `processor/onnxredactionprocessor/internal/metricsmerge/merge.go`**

In `ResolveCollisions` under `case pmetric.MetricTypeExponentialHistogram:`:
- Add `target.SetZeroCount(target.ZeroCount() + dp.ZeroCount())`.
- Add helper `mergeExpBuckets(dst, src pmetric.ExponentialHistogramDataPointBuckets)` that merges `src` into `dst` when `target.Scale() == dp.Scale()` (handling offset alignment by expanding the bucket slice range `[minOffset, maxEnd)` if offsets or lengths differ).

- [ ] **Step 4: Run `metricsmerge` tests to verify pass**

Run: `cd processor/onnxredactionprocessor && go test -race -v ./internal/metricsmerge`
Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add processor/onnxredactionprocessor/internal/metricsmerge/
git commit -m "fix(metricsmerge): merge ZeroCount and Positive/Negative buckets for ExponentialHistogram collisions"
```

---

### Task 5: Processor Signal Coverage, BIO Ordering, Arena Sizing & End-to-End Golden Corpus Tests

**Files:**
- Modify: `processor/onnxredactionprocessor/processor.go:35-401`
- Modify: `processor/onnxredactionprocessor/processor_test.go`
- Modify: `processor/onnxredactionprocessor/corpus_test.go:44-192`

**Interfaces:**
- Consumes: `dedup.TargetSpanStatusMessage`, `coalesce.WithAllowedLabels`, `coalesce.WithLogger`, `(*telemetry).Close()`
- Produces: Complete `redactionProcessor` enforcing canonical BIO ordering, `cfg.Entities`, `span.Status().Message()`, metric `Exemplars()`, `maxNestingDepth` scrubbing, conditional metric collision merging, and verified golden corpus pipeline execution.

- [ ] **Step 1: Write failing tests in `processor_test.go` and `corpus_test.go`**

1. In `processor/onnxredactionprocessor/processor_test.go`, add:
   - `TestProcessor_RealONNXEngineAndEntitiesConfig`: Creates a processor via `newProcessor(cfg, &set.TelemetrySettings, nil, nil)` using `testdata/tiny_ner.onnx` and `testdata/tokenizer.json` with `cfg.Entities = []string{"EMAIL"}`. Sends a log record containing `"token_b-person token_b-email"` and asserts `"token_b-person [REDACTED_EMAIL]"` (proving both `defaultIDToBIO` class index 2 = `B-EMAIL` and `cfg.Entities` filtering work end-to-end).
   - `TestProcessor_SpanStatusAndMetricExemplarsAndMultiTimestampGauge`:
     - Verifies `span.Status().Message()` containing `"alice@example.com"` is redacted to `"[REDACTED_EMAIL]"`.
     - Verifies `dp.Exemplars().At(0).FilteredAttributes()` containing `"John Doe"` is redacted to `"[REDACTED_PERSON]"`.
     - Verifies a `Gauge` metric with 2 datapoints at `ts1` and `ts2` having clean attributes (`host: "srv1"`) preserves **both** datapoints (`Len() == 2`) after `ConsumeMetrics`.
   - `TestProcessor_MaxNestingDepth`: Creates a map nested to depth 34 with string `"secret"` under `on_error: drop` and asserts the string beyond `maxNestingDepth` is scrubbed to `[REDACTED_UNINSPECTED]`.
2. In `processor/onnxredactionprocessor/corpus_test.go`:
   - Update `TestCorpus_GoldenGatedPrecisionRecall` to feed each positive and negative sample into `proc.ConsumeLogs(context.Background(), ld)` and assert that positive log bodies are mutated (`outBody != p.Text`) and negative log bodies are unchanged (`outBody == n.Text`), while computing `tp1/fp1/fn1` and `tp2/fp2/fn2` from spans recorded by `corpusDetector` during `proc.ConsumeLogs`.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestProcessor_RealONNXEngineAndEntitiesConfig|TestProcessor_SpanStatusAndMetricExemplarsAndMultiTimestampGauge|TestProcessor_MaxNestingDepth" .`
Expected: FAIL (`B-EMAIL` misclassified as `I-PERSON`, `PERSON` not filtered by `cfg.Entities`, `span.Status().Message()` unredacted, clean multi-timestamp `Gauge` collapsed to 1 datapoint).

- [ ] **Step 3: Implement fixes in `processor/onnxredactionprocessor/processor.go`**

1. Fix `defaultIDToBIO` (lines 35–46) to group `B-*` at `1..9` and `I-*` at `10..18`:
   ```go
   var defaultIDToBIO = []string{
   	"O",
   	"B-PERSON", "B-EMAIL", "B-PHONE", "B-LOCATION", "B-CREDIT_CARD", "B-SSN", "B-IBAN", "B-MEDICAL_ID", "B-PATH_USER",
   	"I-PERSON", "I-EMAIL", "I-PHONE", "I-LOCATION", "I-CREDIT_CARD", "I-SSN", "I-IBAN", "I-MEDICAL_ID", "I-PATH_USER",
   }
   ```
2. In `newProcessor`:
   - Build `allowed [10]bool` from `cfg.Entities` (if `len(cfg.Entities) == 0`, enable all `1..9`).
   - Filter a copy of `defaultIDToBIO` against `allowed` before calling `decode.NewDecoder`.
   - Pass `cfg.Inference.WindowTokens` and `len(defaultIDToBIO)` into `arena.NewPool(cfg.Inference.MaxTensorBatchSize, cfg.Inference.WindowTokens, len(defaultIDToBIO))`.
   - Pass `coalesce.WithAllowedLabels(allowed)` and `coalesce.WithLogger(set.Logger)` into `coalesce.New`.
3. In `Shutdown`: call `p.telemetry.Close()`.
4. Populate `Signal` on all `dedup.MutationTarget` instances created in `processor.go`.
5. In `processTraces`: if `span.Status().Message() != ""`, append a `coalesce.Item` with `Key: "status.message"`, `Text: span.Status().Message()`, and `Target: dedup.MutationTarget{Kind: dedup.TargetSpanStatusMessage, Span: span, Signal: "traces"}`.
6. In `collectDataPointCandidates`: for `Gauge`, `Sum`, `Histogram`, and `ExponentialHistogram`, iterate `dp.Exemplars()` and call `p.collectMapCandidatesWithMetric("metrics", dp.Exemplars().At(e).FilteredAttributes(), items, name)`.
7. Convert `collectMapCandidates*` and `collectValueCandidates*` to methods on `*redactionProcessor`. When `depth > maxNestingDepth`: call `p.telemetry.RecordSkipped("max_depth", 1)`, and if `p.cfg.OnError == types.ErrorModeDrop`, recursively scrub any `ValueTypeStr` at/beyond `maxNestingDepth` to `types.RedactedUninspected` and call `p.telemetry.RecordScrubbed(signal, "max_depth")`.
8. In `processMetrics`: before calling `p.coalescer.Submit(ctx, items)`, snapshot the attribute values (or track modified metrics via a per-metric flag / pre-and-post canonical hash comparison) so `metricsmerge.ResolveCollisions(m)` is only executed on metrics whose datapoint attributes actually changed during redaction.

- [ ] **Step 4: Run full test suite to verify Task 5**

Run: `make test && make lint`
Expected: PASS across all 14 packages.

- [ ] **Step 5: Commit Task 5**

```bash
git add processor/onnxredactionprocessor/processor.go processor/onnxredactionprocessor/processor_test.go processor/onnxredactionprocessor/corpus_test.go
git commit -m "fix(processor): align BIO taxonomy, enforce entity allowlist, expand signal coverage, and guard Gauge collisions"
```

---

### Task 6: Documentation Audit Fixes, Australian English Validation & Pull Request

**Files:**
- Modify: `README.md:3,107`
- Modify: `docs/howto.md:38,82,128,224,366`
- Modify: `docs/howto/macos.md:20,143-167`

**Interfaces:**
- Consumes: `scripts/validate_australian_english.py`
- Produces: Synchronised documentation and open GitHub Pull Request.

- [ ] **Step 1: Update Go version prerequisites, linker flags, model filenames, and macOS verification guide**

1. In `README.md`:
   - Line 3: change badge from `Go-1.22%2B-blue.svg` to `Go-1.27%2B-blue.svg`.
   - Line 107: change `Version 1.22 or higher.` to `Version 1.27 or higher.`
2. In `docs/howto.md`:
   - Line 38: change `Go 1.22 or newer (1.27 recommended).` to `Go 1.27 or newer.`
   - Line 82: change `-rpath $(pwd)/.ort/lib` to `-Wl,-rpath,$(pwd)/.ort/lib`.
   - Line 128: change `sha256sum ./models/ner-int8/model.onnx` to `sha256sum ./models/ner-int8/model_int8.onnx`.
   - Lines 224 and 366: change `/etc/otel/models/model.onnx` to `/etc/otel/models/tiny_ner.onnx`.
3. In `docs/howto/macos.md`:
   - Line 20: change `Go Compiler (1.22+):` to `Go Compiler (1.27+):`.
   - Lines 143–167: clarify that when running with the bundled synthetic test fixture (`testdata/tiny_ner.onnx`), the deterministic trigger tokens `token_b-person`, `token_b-email`, and `/Users/token_b-path_user/.ssh/id_rsa` demonstrate live redaction (`[REDACTED_PERSON]`, `[REDACTED_EMAIL]`, `/Users/[REDACTED_PATH_USER]/.ssh/id_rsa`), whereas natural-language inputs (`Alice Smith`, `alice@example.com`) require loading a fine-tuned model exported via `scripts/export_and_quantize.py`.

- [ ] **Step 2: Run Australian English validator, linters, tests, and benchmarks**

Run: `make fmt && make lint && make test && make bench`
Expected:
- `PASSED: All 12 markdown file(s) adhere to Australian English spelling.`
- `0 issues.`
- All 14 Go packages pass `-race` tests and benchmarks.

- [ ] **Step 3: Commit documentation updates, push branch, and create Pull Request**

```bash
git add README.md docs/howto.md docs/howto/macos.md docs/superpowers/plans/2026-10-24-gatekeeper-remediation.md
git commit -m "docs: align Go 1.27 requirement, model paths, and macOS synthetic verification payload"
git push -u origin fix/gatekeeper-remediation
gh pr create --title "fix: remediate pre-production gatekeeper review findings" --body "Remediates all validated P0, P1, P2, and documentation findings from the 6-pillar gatekeeper review."
```
