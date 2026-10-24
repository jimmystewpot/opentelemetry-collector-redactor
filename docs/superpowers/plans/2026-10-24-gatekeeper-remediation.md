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

- **Hyphenated and underscored tokens under `BertPreTokenizer` (`alice-smith`, `alice_smith`):** `-` and `_` must split as individual punctuation tokens so production BERT vocabularies match subwords rather than collapsing hyphenated PII (SSNs, phone numbers, credit cards, hyphenated names) into `[UNK]`.
- **Dry-run immutability across pre-inference scrub paths (`dry_run: true`, `on_error: drop`):** Strings exceeding `maxNestingDepth` (32), containing invalid UTF-8, or triggering tokeniser errors must record skip/error telemetry without mutating telemetry values to `[REDACTED_UNINSPECTED]` when `dry_run: true`.
- **Expanding Unicode lowercasing runes (`Ⱥ` U+023A → `ⱥ` U+2C65) exceeding the 256-byte stack buffer:** A 100-rune `Ⱥ` word (200 source bytes → 300 lowercased bytes) with `maxInputChar >= 256` must check encoded rune width before writing to `buf [256]byte`, fall back cleanly to `tokenizeWordSlow`, and map lowercased byte offsets back to valid UTF-8 rune boundaries in `text`.
- **Concurrent `Engine.Close()` during active `Engine.DetectBatch()`:** Calling `Close()` while a slow batch is executing in CGo must wait for `ort_shim_run_batch` to return before freeing the `OrtSession`, without self-deadlocking when returning `runOpts` to `runOptsPool`.
- **Multi-timestamp unredacted Gauge metrics:** A `pmetric.Metric` of type `Gauge` containing multiple data points for the same series across distinct timestamps must not collapse into a single data point when zero attributes in that metric were modified by redaction.

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

- [x] **Step 1: Write failing tests in `processor/onnxredactionprocessor/internal/ort/engine_test.go` for concurrent `NewEngine` init, concurrent `DetectBatch` + `Close()`, and `SpanScratch` reuse**

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

- [x] **Step 2: Run test to verify behaviour under `-race`**

Run: `cd processor/onnxredactionprocessor && go test -race -v -run TestEngine_ConcurrentCloseDuringInference ./internal/ort`
Expected: Race detector or unprotected concurrent access on `e.session` between `DetectBatch` and `Close`.

- [x] **Step 3: Harden `ort_shim.c` and `engine.go`, remove dead code in `arena`/`types`, and deduplicate flags in `Makefile`**

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

- [x] **Step 4: Run tests and linter to verify Task 1**

Run: `make test && make lint`
Expected: PASS with 0 linker warnings and 0 linter issues.

- [x] **Step 5: Commit Task 1**

```bash
git add Makefile processor/onnxredactionprocessor/internal/ort/ processor/onnxredactionprocessor/internal/arena/ processor/onnxredactionprocessor/internal/types/
git commit -m "fix(ort): synchronize CGo session lifecycle, validate 3D output shape, and remove dead code"
```

**Code Review & Completion Criteria:**
- `ort_shim_init` is guarded by `sync.Once` (`initOnce`), eliminating CGo global `g_ort` data races during concurrent `NewEngine` calls.
- `ort_shim_run_batch` validates `num_dims == 3 && dims[0] == batch_size && dims[1] == seq_len && dims[2] > 0` before computing `total_elements` or calling `memcpy`, releasing `shape_info` and `out_tensors[0]` on error.
- `DetectBatch` holds `e.mu.RLock()` across CGo execution and releases `runOpts` via non-blocking channel send inside `defer` before calling `e.mu.RUnlock()`, with zero nested `e.mu.Lock()` calls.
- `go test -race ./internal/ort` passes with 0 race warnings and 0 duplicate linker flag warnings.

---

### Task 2: Zero-Allocation WordPiece Lowercasing & Subword Prefixing

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:125-242`
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`

**Interfaces:**
- Consumes: `types.Offset`
- Produces: Zero-allocation `(*wordPieceTokenizer).Encode(text string, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset, error)`

- [x] **Step 1: Write failing allocation test and benchmark in `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`**

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

- [x] **Step 2: Run test to verify it fails**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestWordPieceTokenizer_ZeroAllocations ./internal/tokenizer`
Expected: FAIL (`allocs > 0` due to `strings.ToLower` and `t.prefix + subStr`).

- [x] **Step 3: Implement stack-buffered lowercasing and subword prefixing in `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go`**

In `Encode` and `tokenizeWord`:
- Check word byte length `wordEnd - wordStart` (or lowercased length) against `t.maxInputChar` first.
- Use a stack buffer `var buf [256]byte` where `prefixLen := len(t.prefix)` is reserved at the start (`buf[prefixLen:]` holds the lowercased word bytes; ASCII `'A'..'Z'` is lowercased inline via `b + ('a' - 'A')`, falling back to `utf8.EncodeRune(buf[n:], unicode.ToLower(r))` for non-ASCII runes).
- For whole-word lookup, index `t.vocab[string(buf[prefixLen:prefixLen+wordLen])]`.
- For subword lookup when `subStart > 0`, copy `t.prefix` into `buf[prefixLen+subStart-prefixLen : prefixLen+subStart]`, look up `t.vocab[string(buf[subStart : prefixLen+subEnd])]`, and restore the overwritten bytes before advancing `subStart`.

- [x] **Step 4: Run tokeniser tests and verify 0 allocations**

Run: `cd processor/onnxredactionprocessor && go test -race -v ./internal/tokenizer`
Expected: PASS (`TestWordPieceTokenizer_ZeroAllocations` passes with 0 allocs; note `-race` may instrument map accesses so gate `testing.AllocsPerRun` or run without `-race` to confirm 0 allocs, and ensure `make test` passes).

- [x] **Step 5: Commit Task 2**

```bash
git add processor/onnxredactionprocessor/internal/tokenizer/
git commit -m "perf(tokenizer): eliminate heap allocations in WordPiece lowercasing and subword lookup"
```

**Code Review & Completion Criteria:**
- `TestWordPieceTokenizer_ZeroAllocations` passes with `0 allocs/op` when pre-allocated `dstIDs` and `dstOffsets` slices are supplied.
- Stack buffer `buf [256]byte` reserves `prefixLen` bytes at the front so `"##"` subword lookups overwrite and restore `savedPrefix` in-place without heap string concatenation.
- `TestWordPieceTokenizer_FastSlowDifferential` and `TestWordPieceTokenizer_RuneLengthChangeOffsets` pass under `-race`.

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

- [x] **Step 1: Write failing tests in `dedup_test.go` and `coalesce_test.go`**

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

- [x] **Step 2: Run tests to verify failure**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestCoalescer_AllowedLabelsFiltering ./internal/coalesce`
Expected: FAIL (`WithAllowedLabels` undefined).

- [x] **Step 3: Implement `TargetSpanStatusMessage`, `WithAllowedLabels`, `WithLogger`, event-driven cancellation, and `telemetry.Close()`**

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

- [x] **Step 4: Run tests to verify Task 3 passes**

Run: `make test && make lint`
Expected: PASS.

- [x] **Step 5: Commit Task 3**

```bash
git add processor/onnxredactionprocessor/internal/dedup/ processor/onnxredactionprocessor/internal/coalesce/ processor/onnxredactionprocessor/telemetry.go
git commit -m "feat(coalesce): add entity allowlist filtering, event-driven cancellation, panic logging, and signal scrub metrics"
```

**Code Review & Completion Criteria:**
- No `time.NewTicker` polling remains in `coalesce.executeBatch`; caller cancellation is monitored via event-driven `select` on `req.ctx.Done()`, `stopMonitor`, and `ctx.Done()`.
- `WithAllowedLabels` filters spans by `types.Label` before span merging, and `WithLogger` logs recovered panics at `ErrorLevel` with `zap.Any("panic", r)` and `zap.Stack("stacktrace")`.
- `telemetry.Close()` unregisters `t.reg` idempotently (`t.reg = nil` after `Unregister()`).

---

### Task 4: Metric Collision Resolution — ExponentialHistogram Bucket & ZeroCount Merging

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge.go:217-255`
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go:158-190`

**Interfaces:**
- Consumes: `pmetric.ExponentialHistogramDataPoint`, `pmetric.ExponentialHistogramDataPointBuckets`
- Produces: Complete `ResolveCollisions` for `MetricTypeExponentialHistogram` merging `ZeroCount`, `Positive`, and `Negative` buckets.

- [x] **Step 1: Update `TestResolveCollisions_ExponentialHistogram` in `processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go` to assert `ZeroCount` and bucket merging**

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

- [x] **Step 2: Run test to verify it fails**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestResolveCollisions_ExponentialHistogram ./internal/metricsmerge`
Expected: FAIL (`ZeroCount` is `2` instead of `3`, `Positive().BucketCounts()` is `[3, 5]` instead of `[5, 7]`).

- [x] **Step 3: Implement `ZeroCount` and bucket merging in `processor/onnxredactionprocessor/internal/metricsmerge/merge.go`**

In `ResolveCollisions` under `case pmetric.MetricTypeExponentialHistogram:`:
- Add `target.SetZeroCount(target.ZeroCount() + dp.ZeroCount())`.
- Add helper `mergeExpBuckets(dst, src pmetric.ExponentialHistogramDataPointBuckets)` that merges `src` into `dst` when `target.Scale() == dp.Scale()` (handling offset alignment by expanding the bucket slice range `[minOffset, maxEnd)` if offsets or lengths differ).

- [x] **Step 4: Run `metricsmerge` tests to verify pass**

Run: `cd processor/onnxredactionprocessor && go test -race -v ./internal/metricsmerge`
Expected: PASS.

- [x] **Step 5: Commit Task 4**

```bash
git add processor/onnxredactionprocessor/internal/metricsmerge/
git commit -m "fix(metricsmerge): merge ZeroCount and Positive/Negative buckets for ExponentialHistogram collisions"
```

**Code Review & Completion Criteria:**
- Colliding `ExponentialHistogram` data points with matching `Scale()` sum `Count()`, `Sum()`, `ZeroCount()`, and both `Positive()` and `Negative()` bucket arrays across aligned or offset-shifted ranges `[minOffset, maxEnd)`.
- `Min` and `Max` extrema and exemplar transfers are preserved without mutating non-colliding series.

---

### Task 5: Processor Signal Coverage, BIO Ordering, Arena Sizing & End-to-End Golden Corpus Tests

**Files:**
- Modify: `processor/onnxredactionprocessor/processor.go:35-401`
- Modify: `processor/onnxredactionprocessor/processor_test.go`
- Modify: `processor/onnxredactionprocessor/corpus_test.go:44-192`

**Interfaces:**
- Consumes: `dedup.TargetSpanStatusMessage`, `coalesce.WithAllowedLabels`, `coalesce.WithLogger`, `(*telemetry).Close()`
- Produces: Complete `redactionProcessor` enforcing canonical BIO ordering, `cfg.Entities`, `span.Status().Message()`, metric `Exemplars()`, `maxNestingDepth` scrubbing, conditional metric collision merging, and verified golden corpus pipeline execution.

- [x] **Step 1: Write failing tests in `processor_test.go` and `corpus_test.go`**

1. In `processor/onnxredactionprocessor/processor_test.go`, add:
   - `TestProcessor_RealONNXEngineAndEntitiesConfig`: Creates a processor via `newProcessor(cfg, &set.TelemetrySettings, nil, nil)` using `testdata/tiny_ner.onnx` and `testdata/tokenizer.json` with `cfg.Entities = []string{"EMAIL"}`. Sends a log record containing `"token_b-person token_b-email"` and asserts `"token_b-person [REDACTED_EMAIL]"` (proving both `defaultIDToBIO` class index 2 = `B-EMAIL` and `cfg.Entities` filtering work end-to-end).
   - `TestProcessor_SpanStatusAndMetricExemplarsAndMultiTimestampGauge`:
     - Verifies `span.Status().Message()` containing `"alice@example.com"` is redacted to `"[REDACTED_EMAIL]"`.
     - Verifies `dp.Exemplars().At(0).FilteredAttributes()` containing `"John Doe"` is redacted to `"[REDACTED_PERSON]"`.
     - Verifies a `Gauge` metric with 2 datapoints at `ts1` and `ts2` having clean attributes (`host: "srv1"`) preserves **both** datapoints (`Len() == 2`) after `ConsumeMetrics`.
   - `TestProcessor_MaxNestingDepth`: Creates a map nested to depth 34 with string `"secret"` under `on_error: drop` and asserts the string beyond `maxNestingDepth` is scrubbed to `[REDACTED_UNINSPECTED]`.
2. In `processor/onnxredactionprocessor/corpus_test.go`:
   - Update `TestCorpus_GoldenGatedPrecisionRecall` to feed each positive and negative sample into `proc.ConsumeLogs(context.Background(), ld)` and assert that positive log bodies are mutated (`outBody != p.Text`) and negative log bodies are unchanged (`outBody == n.Text`), while computing `tp1/fp1/fn1` and `tp2/fp2/fn2` from spans recorded by `corpusDetector` during `proc.ConsumeLogs`.

- [x] **Step 2: Run tests to verify failure**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestProcessor_RealONNXEngineAndEntitiesConfig|TestProcessor_SpanStatusAndMetricExemplarsAndMultiTimestampGauge|TestProcessor_MaxNestingDepth" .`
Expected: FAIL (`B-EMAIL` misclassified as `I-PERSON`, `PERSON` not filtered by `cfg.Entities`, `span.Status().Message()` unredacted, clean multi-timestamp `Gauge` collapsed to 1 datapoint).

- [x] **Step 3: Implement fixes in `processor/onnxredactionprocessor/processor.go`**

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

- [x] **Step 4: Run full test suite to verify Task 5**

Run: `make test && make lint`
Expected: PASS across all 14 packages.

- [x] **Step 5: Commit Task 5**

```bash
git add processor/onnxredactionprocessor/processor.go processor/onnxredactionprocessor/processor_test.go processor/onnxredactionprocessor/corpus_test.go
git commit -m "fix(processor): align BIO taxonomy, enforce entity allowlist, expand signal coverage, and guard Gauge collisions"
```

**Code Review & Completion Criteria:**
- `defaultIDToBIO` indices `1..9` are `B-*` and `10..18` are `I-*`, matching `docs/models/model-decision-record.md` and `scripts/generate_test_onnx.py`.
- `span.Status().Message()` and `dp.Exemplars().At(e).FilteredAttributes()` are inspected and redacted across all supported signals and metric types.
- `metricsmerge.ResolveCollisions(m)` is skipped when a metric's canonical datapoint attribute hash is unchanged, preserving multi-timestamp unredacted `Gauge` series.
- `TestCorpus_GoldenGatedPrecisionRecall` routes all positive and negative samples through `proc.ConsumeLogs` and meets Tier 1 (Recall >= 0.98, Precision >= 0.92) and Tier 2 (Recall >= 0.92, Precision >= 0.85) gates with 0 negative false positives.

---

### Task 6: Documentation Audit Fixes, Australian English Validation & Pull Request

**Files:**
- Modify: `README.md:3,107`
- Modify: `docs/howto.md:38,82,128,224,366`
- Modify: `docs/howto/macos.md:20,143-167`

**Interfaces:**
- Consumes: `scripts/validate_australian_english.py`
- Produces: Synchronised documentation and open GitHub Pull Request.

- [x] **Step 1: Update Go version prerequisites, linker flags, model filenames, and macOS verification guide**

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

- [x] **Step 2: Run Australian English validator, linters, tests, and benchmarks**

Run: `make fmt && make lint && make test && make bench`
Expected:
- `PASSED: All 12 markdown file(s) adhere to Australian English spelling.`
- `0 issues.`
- All 14 Go packages pass `-race` tests and benchmarks.

- [x] **Step 3: Commit documentation updates, push branch, and create Pull Request**

```bash
git add README.md docs/howto.md docs/howto/macos.md docs/superpowers/plans/2026-10-24-gatekeeper-remediation.md
git commit -m "docs: align Go 1.27 requirement, model paths, and macOS synthetic verification payload"
git push -u origin fix/gatekeeper-remediation
gh pr create --title "fix: remediate pre-production gatekeeper review findings" --body "Remediates all validated P0, P1, P2, and documentation findings from the 6-pillar gatekeeper review."
```

**Code Review & Completion Criteria:**
- All markdown documentation (`README.md`, `docs/howto.md`, `docs/howto/*.md`, `docs/models/*.md`, `docs/superpowers/**/*.md`) passes `python3 scripts/validate_australian_english.py docs/ README.md` (`make validate-spelling`).
- Go prerequisite badges and text consistently specify `1.27+`.

---

### Task 7: Preserve `BertPreTokenizer` Punctuation Splitting & Migrate Synthetic Trigger Tokens (PR2-1)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:381-384`
- Modify: `scripts/generate_test_onnx.py:12-15,30-35,94-99,188-205,252-284`
- Modify: `processor/onnxredactionprocessor/testdata/tokenizer.json`
- Modify: `processor/onnxredactionprocessor/testdata/checksums.txt:3`
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go:17,94-116`
- Modify: `processor/onnxredactionprocessor/processor_test.go:500-510`
- Modify: `config.yaml:20`
- Modify: `README.md:177`
- Modify: `docs/howto.md:227`
- Modify: `docs/howto/macos.md:146-172`

**Interfaces:**
- Consumes: `NewWordPieceTokenizer(tokenizerJSONPath, expectedSHA256 string) (Tokenizer, error)`
- Produces:
  - `isPunctOrCJK(r rune) bool` splitting `-` and `_` as standalone punctuation tokens per `BertPreTokenizer`
  - `python3 scripts/generate_test_onnx.py --tokenizer-only` CLI flag with lazy `numpy`/`onnx` imports inside `make_tiny_ner_model()` and `make_slow_loop_model()`
  - Synthetic trigger tokens `tokenbperson`..`tokenbpathuser` (`200..208`) and `tokeniperson`..`tokenipathuser` (`210..218`), plus `"-"`: `367` and `"_"`: `368` in `testdata/tokenizer.json` (`SHA-256: 2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884`)

- [x] **Step 1: Write failing test in `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go` for hyphen and underscore splitting**

```go
func TestWordPieceTokenizer_HyphenAndUnderscoreSplitting(t *testing.T) {
	tok, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tok.Close() }()

	// "alice-smith" -> [CLS](101), "alice"(314), "-"(367), "smith"(318), [SEP](102)
	ids, offsets, err := tok.Encode("alice-smith", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []int64{101, 314, 367, 318, 102}, ids)
	assert.Equal(t, []types.Offset{
		{Start: 0, End: 0},
		{Start: 0, End: 5},
		{Start: 5, End: 6},
		{Start: 6, End: 11},
		{Start: 11, End: 11},
	}, offsets)

	// "alice_smith" -> [CLS](101), "alice"(314), "_"(368), "smith"(318), [SEP](102)
	idsUnder, offsetsUnder, err := tok.Encode("alice_smith", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []int64{101, 314, 368, 318, 102}, idsUnder)
	assert.Equal(t, []types.Offset{
		{Start: 0, End: 0},
		{Start: 0, End: 5},
		{Start: 5, End: 6},
		{Start: 6, End: 11},
		{Start: 11, End: 11},
	}, offsetsUnder)
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestWordPieceTokenizer_HyphenAndUnderscoreSplitting ./internal/tokenizer`
Expected: FAIL (`"alice-smith"` is emitted as single `[UNK]` token `100` because `isPunctOrCJK` returns `false` for `'-'` and `'_'`).

- [x] **Step 3: Revert `isPunctOrCJK` exemption and update `scripts/generate_test_onnx.py`, `testdata/tokenizer.json`, checksums, tests, and docs**

1. In `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:381-384`, remove `if r == '_' || r == '-' { return false }` from `isPunctOrCJK(r rune) bool`.
2. In `scripts/generate_test_onnx.py`:
   - Move `import numpy as np` and `import onnx` / `from onnx import helper, TensorProto` inside `make_tiny_ner_model()` and `make_slow_loop_model()` so `--tokenizer-only` runs using only the Python standard library.
   - Add `argparse` support for `--tokenizer-only` in `main()`.
   - Format synthetic trigger keys in `make_tokenizer_json()` as `f"token{label.lower().replace('-', '').replace('_', '')}"` and append `"-"`, `"_"` after `"##net"` in `common_words`.
   - Run `python3 scripts/generate_test_onnx.py --tokenizer-only` to regenerate `processor/onnxredactionprocessor/testdata/tokenizer.json` and update `processor/onnxredactionprocessor/testdata/checksums.txt` (`2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884  tokenizer.json`).
3. Update `validSHA256` in `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go:17` and `tokenizer_sha256` in `config.yaml:20`, `README.md:177`, and `docs/howto.md:227` to `2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884`.
4. Update `TestProcessor_RealONNXEngineAndEntitiesConfig` in `processor/onnxredactionprocessor/processor_test.go:502-509` to send `"tokenbperson tokenbemail tokenbperson-tokenbemail"` and assert `"tokenbperson [REDACTED_EMAIL] tokenbperson-[REDACTED_EMAIL]"`.
5. Update `docs/howto/macos.md:146-172` to use `tokenbperson`, `tokenbemail`, and `/Users/tokenbpathuser/.ssh/id_rsa`.

- [x] **Step 4: Run tests and spelling validation to verify Task 7**

Run: `make test && make validate-spelling`
Expected: PASS.

- [x] **Step 5: Commit Task 7**

```bash
git add processor/onnxredactionprocessor/internal/tokenizer/ processor/onnxredactionprocessor/testdata/ processor/onnxredactionprocessor/processor_test.go scripts/generate_test_onnx.py config.yaml README.md docs/howto.md docs/howto/macos.md
git commit -m "fix(tokenizer): preserve BertPreTokenizer punctuation splitting for hyphens and underscores"
```

**Code Review & Completion Criteria:**
- `isPunctOrCJK(r)` contains zero special-case exemptions for `-` or `_`; all Unicode `P` (Punctuation) and `S` (Symbol) runes split as individual single-rune tokens.
- `python3 scripts/generate_test_onnx.py --tokenizer-only` succeeds on hosts without `onnx` or `numpy` installed, leaves `tiny_ner.onnx` and `slow_loop.onnx` untouched (`git diff --stat` shows zero changes to `.onnx` binaries), and produces `SHA-256: 2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884`.
- All 5 occurrences of `tokenizer.json` SHA-256 (`testdata/checksums.txt`, `tokenizer_test.go`, `config.yaml`, `README.md`, `docs/howto.md`) match `2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884`, and `git grep "094033eb"` returns 0 matches outside `docs/superpowers/`.
- `TestProcessor_RealONNXEngineAndEntitiesConfig` proves both space-separated (`"tokenbperson tokenbemail"`) and hyphen-separated (`"tokenbperson-tokenbemail"`) strings redact only the `EMAIL` span with exact byte offsets (`"tokenbperson [REDACTED_EMAIL] tokenbperson-[REDACTED_EMAIL]"`).

---

### Task 8: Honour `dry_run: true` Across All Pre-Inference Scrubbing Paths & Pre-Collection Metric Hashing (PR2-2)

**Files:**
- Modify: `processor/onnxredactionprocessor/processor.go:298-350,459-466`
- Modify: `processor/onnxredactionprocessor/internal/coalesce/coalesce.go:386-392,431-440`
- Modify: `processor/onnxredactionprocessor/processor_test.go:585-619`
- Modify: `processor/onnxredactionprocessor/internal/coalesce/coalesce_test.go:754-815`

**Interfaces:**
- Consumes: `Config.DryRun bool`, `Coalescer.dryRun bool`, `metricDataPointsHash(m pmetric.Metric, scratch []string) (uint64, []string)`
- Produces: Dry-run-safe `(*redactionProcessor).collectValueCandidatesWithMetric`, single-pass pre-collection hashing in `(*redactionProcessor).processMetrics`, and dry-run-safe `(*Coalescer).executeBatch`

- [x] **Step 1: Write failing tests in `processor_test.go` and `coalesce_test.go` for dry-run immutability and telemetry counters on `maxNestingDepth`, `invalid_utf8`, and `tokenizer_error`**

In `processor/onnxredactionprocessor/processor_test.go` (`TestProcessor_MaxNestingDepth`, importing `sdkmetric "go.opentelemetry.io/otel/sdk/metric"` and `"go.opentelemetry.io/otel/sdk/metric/metricdata"`):
```go
	// Dry-run mode with on_error: drop must NOT scrub deeply nested values,
	// but MUST still record redaction_skipped_total{reason="max_depth"}
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	defer func() { _ = mp.Shutdown(context.Background()) }()

	cfgDry := createDefaultConfig().(*Config)
	cfgDry.Inference.CoalesceWindow = 5 * time.Millisecond
	cfgDry.OnError = types.ErrorModeDrop
	cfgDry.DryRun = true

	setDry := processortest.NewNopSettings(Type)
	setDry.TelemetrySettings.MeterProvider = mp
	procDry, err := newProcessor(cfgDry, &setDry.TelemetrySettings, &mockTokenizer{}, &ruleDetector{})
	require.NoError(t, err)
	defer func() { _ = procDry.Shutdown(context.Background()) }()

	sinkDry := new(consumertest.LogsSink)
	procDry.nextLogs = sinkDry

	ldDry := plog.NewLogs()
	recDry := ldDry.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
	currDry := recDry.Body().SetEmptyMap()
	for i := 0; i < 34; i++ {
		currDry = currDry.PutEmptyMap(fmt.Sprintf("k%d", i))
	}
	currDry.PutStr("secret_key", "secret")

	require.NoError(t, procDry.ConsumeLogs(context.Background(), ldDry))
	require.Len(t, sinkDry.AllLogs(), 1)
	outDry := sinkDry.AllLogs()[0].ResourceLogs().At(0).ScopeLogs().At(0).LogRecords().At(0).Body().Map()
	for i := 0; i < 34; i++ {
		v, ok := outDry.Get(fmt.Sprintf("k%d", i))
		require.True(t, ok)
		outDry = v.Map()
	}
	leafDry, ok := outDry.Get("secret_key")
	require.True(t, ok)
	assert.Equal(t, "secret", leafDry.Str(), "dry_run must preserve deeply nested string")

	var rm metricdata.ResourceMetrics
	require.NoError(t, reader.Collect(context.Background(), &rm))
	var skippedCount, scrubbedCount int64
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			switch m.Name {
			case "redaction_skipped_total":
				for _, dp := range m.Data.(metricdata.Sum[int64]).DataPoints {
					skippedCount += dp.Value
				}
			case "redaction_scrubbed_total":
				for _, dp := range m.Data.(metricdata.Sum[int64]).DataPoints {
					scrubbedCount += dp.Value
				}
			}
		}
	}
	assert.Equal(t, int64(1), skippedCount, "redaction_skipped_total must be recorded in dry_run")
	assert.Equal(t, int64(0), scrubbedCount, "redaction_scrubbed_total must be 0 in dry_run")
```

In `processor/onnxredactionprocessor/internal/coalesce/coalesce_test.go`:
```go
func TestCoalescer_DryRunPreservesInvalidUTF8AndTokenizerError(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	rec := &scrubRecorder{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	c := New(5*time.Millisecond, 2, nil, &errTokenizer{}, chunker, bucketer, fakeDet, pool, 16384, types.ErrorModeDrop, true, WithTelemetry(rec))
	defer func() { _ = c.Close() }()

	badUTF8 := pcommon.NewValueStr("bad\xffutf8")
	tokFail := pcommon.NewValueStr("valid text")

	err := c.Submit(context.Background(), []Item{
		{Signal: "logs", Text: badUTF8.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: badUTF8, Signal: "logs"}},
		{Signal: "logs", Text: tokFail.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: tokFail, Signal: "logs"}},
	})
	assert.NoError(t, err)
	assert.Equal(t, "bad\xffutf8", badUTF8.Str())
	assert.Equal(t, "valid text", tokFail.Str())

	rec.mu.Lock()
	defer rec.mu.Unlock()
	assert.Empty(t, rec.scrubbed, "dry_run must not record scrubbed strings or mutate values")
}
```

- [x] **Step 2: Run tests to verify failure**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestProcessor_MaxNestingDepth|TestCoalescer_DryRunPreservesInvalidUTF8AndTokenizerError" ./...`
Expected: FAIL (`leafDry.Str()` and `badUTF8.Str()` are mutated to `"[REDACTED_UNINSPECTED]"` and `scrubbedCount > 0`).

- [x] **Step 3: Gate pre-inference scrubbing on `!DryRun` and snapshot metric hashes before candidate collection**

1. In `processor/onnxredactionprocessor/processor.go:298-350` (`processMetrics`):
   - Pre-allocate `metricHashes := make([]uint64, 0, md.MetricCount())` and `var scratch []string` before the first `rMetrics` loop.
   - Inside the `for k := 0; k < metrics.Len(); k++` loop, compute `h, scratch = metricDataPointsHash(m, scratch)` and `metricHashes = append(metricHashes, h)` *before* calling `p.collectDataPointCandidates(m, &items)`, and remove the duplicate second pre-Submit loop.
2. In `processor/onnxredactionprocessor/processor.go:462`: change `if p.cfg.OnError == types.ErrorModeDrop` to `if p.cfg.OnError == types.ErrorModeDrop && !p.cfg.DryRun`.
3. In `processor/onnxredactionprocessor/internal/coalesce/coalesce.go:386-392`: call `c.telemetry.RecordError("invalid_utf8")` unconditionally when `!utf8.ValidString(item.Text)`, and gate `item.Target.Set(types.RedactedUninspected)` and `c.telemetry.RecordScrubbed(item.Signal, "invalid_utf8")` on `if c.errorMode == types.ErrorModeDrop && !c.dryRun`.
4. In `processor/onnxredactionprocessor/internal/coalesce/coalesce.go:431-440`: call `c.telemetry.RecordError("tokenizer_error")` unconditionally when `c.tokenizer.Encode` fails, and gate the `trg.Set(types.RedactedUninspected)` + `c.telemetry.RecordScrubbed(trg.Signal, "tokenizer_error")` loop on `if c.errorMode == types.ErrorModeDrop && !c.dryRun`.

- [x] **Step 4: Run tests and linter to verify Task 8**

Run: `make test && make lint`
Expected: PASS.

- [x] **Step 5: Commit Task 8**

```bash
git add processor/onnxredactionprocessor/processor.go processor/onnxredactionprocessor/processor_test.go processor/onnxredactionprocessor/internal/coalesce/coalesce.go processor/onnxredactionprocessor/internal/coalesce/coalesce_test.go
git commit -m "fix(processor): honour dry_run mode across max_depth and pre-inference scrub paths"
```

**Code Review & Completion Criteria:**
- Every path that calls `Set(types.RedactedUninspected)` or `p.scrubValue` across `processor.go` and `coalesce.go` is guarded by `!DryRun` (`!p.cfg.DryRun` / `!c.dryRun`).
- `redaction_skipped_total{reason="max_depth"}` and `redaction_errors_total{reason="invalid_utf8"|"tokenizer_error"}` are still recorded when `dry_run: true`, while `redaction_scrubbed_total` remains `0`.
- `processMetrics` computes `metricDataPointsHash` before `p.collectDataPointCandidates` in a single pre-Submit loop, eliminating the redundant traversal and capturing pre-scrubbing state.

---

### Task 9: Expanding Lowercase Rune Buffer Guard & Byte-Indexed Slow-Path Offset Mapping (PR2-3)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:212-236,296-306,319-379`
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go:170-220`

**Interfaces:**
- Consumes: `unicode.ToLower(r rune) rune`, `utf8.RuneLen(r rune) int`, `utf8.EncodeRune(p []byte, r rune) int`
- Produces: Buffer-safe `(*wordPieceTokenizer).tokenizeWord` (`textOffsets [257]uint16`) and byte-offset-accurate `(*wordPieceTokenizer).tokenizeWordSlow` (`textOffsets []int` of length `len(word)+1`)

- [x] **Step 1: Write failing tests in `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go` for expanding lowercase runes (`Ⱥ` → `ⱥ`) at both `maxInputChar = 256` and `maxInputChar = 350`, plus unconditional fast/slow offset parity**

```go
func TestWordPieceTokenizer_ExpandingLowercaseRunes(t *testing.T) {
	tokIface, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tokIface.Close() }()

	tok := tokIface.(*wordPieceTokenizer)
	// 'Ⱥ' (U+023A, 2 UTF-8 bytes) lowercases to 'ⱥ' (U+2C65, 3 UTF-8 bytes)
	tok.vocab["ⱥ"] = 500
	tok.vocab["##ⱥ"] = 501

	// 100 repetitions of 'Ⱥ': 200 bytes in source text, expands to 300 bytes in lowercase (> 256-byte stack buf)
	word := strings.Repeat("Ⱥ", 100)

	// Case 1: maxInputChar = 256 (passes initial 200-byte rawLen check, overflows 256-byte stack buf,
	// falls back to tokenizeWordSlow, exceeds 256-byte lowercased limit, and emits [UNK] without panicking)
	tok.maxInputChar = 256
	idsUnk, offsetsUnk, err := tok.Encode(word, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []int64{101, 100, 102}, idsUnk)
	assert.Equal(t, []types.Offset{
		{Start: 0, End: 0},
		{Start: 0, End: 200},
		{Start: 200, End: 200},
	}, offsetsUnk)

	// Case 2: maxInputChar = 350 (falls back to tokenizeWordSlow and segments into 100 subwords
	// with exact 2-byte source offsets rather than 3-byte lowercased offsets)
	tok.maxInputChar = 350
	ids, offsets, err := tok.Encode(word, nil, nil)
	require.NoError(t, err)
	require.Len(t, ids, 102)
	assert.Equal(t, int64(500), ids[1])
	assert.Equal(t, int64(501), ids[2])

	for i := 1; i <= 100; i++ {
		expectedStart := (i - 1) * 2
		expectedEnd := i * 2
		assert.Equal(t, types.Offset{Start: expectedStart, End: expectedEnd}, offsets[i], "token %d offset mismatch", i)
		assert.True(t, utf8.RuneStart(word[offsets[i].Start]))
	}
}
```
And in `TestWordPieceTokenizer_FastSlowDifferential`, add `strings.Repeat("Ⱥ", 10)` to `words` and remove the `if !hasChange` condition so `assert.Equal(t, slowOffsets, fastOffsets)` runs unconditionally for every word including `"İzmir"`, `"Kelvin"`, and `strings.Repeat("Ⱥ", 10)`.

- [x] **Step 2: Run tests to verify failure**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestWordPieceTokenizer_ExpandingLowercaseRunes|TestWordPieceTokenizer_FastSlowDifferential" ./internal/tokenizer`
Expected: FAIL (`tokenizeWordSlow` emits lowercased byte offsets `[0,3]`, `[3,6]`, ..., `[297,300]` which exceed `len(word) == 200` and fail `utf8.RuneStart`, and `TestWordPieceTokenizer_FastSlowDifferential` fails on `"İzmir"` and `"Kelvin"`).

- [x] **Step 3: Check exact encoded rune width in `tokenizeWord` and map byte-indexed source offsets in `tokenizeWordSlow`**

1. In `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:214-235` (`tokenizeWord`):
   - Widen `var textOffsets [257]uint8` to `var textOffsets [257]uint16`.
   - Check `encSz := utf8.RuneLen(rLow)` before writing to `textOffsets` or `buf`:
     ```go
     	if !isASCII {
     		srcRel := 0
     		dstRel := 0
     		for srcRel < rawLen {
     			r, sz := utf8.DecodeRuneInString(text[wordStart+srcRel : wordEnd])
     			rLow := unicode.ToLower(r)
     			encSz := utf8.RuneLen(rLow)
     			if encSz <= 0 || prefixLen+dstRel+encSz > len(buf) {
     				return t.tokenizeWordSlow(text, wordStart, wordEnd, dstIDs, dstOffsets)
     			}
     			textOffsets[dstRel] = uint16(srcRel) //nolint:gosec // srcRel <= rawLen <= 256
     			utf8.EncodeRune(buf[prefixLen+dstRel:], rLow)
     			if encSz != sz {
     				hasRuneLenChange = true
     			}
     			srcRel += sz
     			dstRel += encSz
     		}
     		textOffsets[dstRel] = uint16(srcRel) //nolint:gosec // srcRel <= rawLen <= 256
     		lowercasedLen = dstRel
     	}
     ```
2. In `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:319-379` (`tokenizeWordSlow`):
   - Lowercase rune-by-rune with `unicode.ToLower(r)` and build a **byte-indexed** `textOffsets` slice of length `len(word) + 1` (appending `srcRel` once for each of the `encSz` encoded UTF-8 bytes of `rLow`, plus `rawLen` at the end):
     ```go
     func (t *wordPieceTokenizer) tokenizeWordSlow(text string, wordStart, wordEnd int, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset) {
     	rawLen := wordEnd - wordStart
     	var wordBuf strings.Builder
     	wordBuf.Grow(rawLen)
     	textOffsets := make([]int, 0, rawLen+1)
     	for srcRel := 0; srcRel < rawLen; {
     		r, sz := utf8.DecodeRuneInString(text[wordStart+srcRel : wordEnd])
     		rLow := unicode.ToLower(r)
     		var rBuf [utf8.UTFMax]byte
     		encSz := utf8.EncodeRune(rBuf[:], rLow)
     		for i := 0; i < encSz; i++ {
     			textOffsets = append(textOffsets, srcRel)
     		}
     		wordBuf.Write(rBuf[:encSz])
     		srcRel += sz
     	}
     	textOffsets = append(textOffsets, rawLen)
     	word := wordBuf.String()
     ```
   - When appending subword offsets in `tokenizeWordSlow`, emit `types.Offset{Start: wordStart + textOffsets[subStart], End: wordStart + textOffsets[subEnd]}`.

- [x] **Step 4: Run full test suite, linter, and benchmarks to verify Task 9**

Run: `make fmt && make lint && make test && make bench`
Expected: PASS with 0 issues and 0 allocations on the fast path.

- [x] **Step 5: Commit Task 9 and push to PR #2**

```bash
git add processor/onnxredactionprocessor/internal/tokenizer/ docs/superpowers/specs/2026-10-24-gatekeeper-remediation-design.md docs/superpowers/plans/2026-10-24-gatekeeper-remediation.md
git commit -m "fix(tokenizer): check encoded rune capacity and preserve source byte offsets in slow path"
git push
```

**Code Review & Completion Criteria:**
- In `tokenizeWord`, `encSz := utf8.RuneLen(rLow)` and `if encSz <= 0 || prefixLen+dstRel+encSz > len(buf)` execute strictly before `textOffsets[dstRel]` assignment and `utf8.EncodeRune(buf[prefixLen+dstRel:], rLow)`.
- In `tokenizeWordSlow`, `len(textOffsets) == len(word) + 1`, indexed by lowercased byte position `subStart` and `subEnd`, preventing both out-of-bounds panics and mid-rune byte offsets on multi-byte expanding (`Ⱥ` → `ⱥ`) or shrinking (`İ` → `i`, `K` → `k`) runes.
- `TestWordPieceTokenizer_FastSlowDifferential` asserts unconditional equality of both `ids` and `offsets` between `tokenizeWord` and `tokenizeWordSlow` across all ASCII and Unicode test cases.
- `TestWordPieceTokenizer_ZeroAllocations` continues to pass with `0 allocs/op`.
