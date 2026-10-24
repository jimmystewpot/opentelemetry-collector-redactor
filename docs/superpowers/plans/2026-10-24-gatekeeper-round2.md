# Gatekeeper Review Round 2 Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all validated P0, P1, P2, and documentation findings from the Round 2 static analysis and memory safety audit using test-driven development.

**Architecture:** Harden span escape widening and post-widening overlap clamping in `internal/mask`; enforce CGo `FLOAT` tensor element type verification, `runtime.KeepAlive`, and decoder class bounds in `internal/ort`, `internal/decode`, and `internal/tokenizer`; bound pooled slice growth, eliminate redundant batch slice allocations, clean up failed initialisation registrations, and harden metric collision merging across `internal/arena`, `internal/coalesce`, `internal/metricsmerge`, `processor.go`, and `README.md`.

**Tech Stack:** Go 1.27, CGo (`onnxruntime_c_api.h` v1.31.0), OpenTelemetry Collector `pdata` v0.163.0, `stretchr/testify`.

**Spec:** `docs/superpowers/specs/2026-10-24-gatekeeper-round2-design.md`

## Global Constraints

- All code must compile and pass `go test -race -count=1 ./...` (`make test`) and `golangci-lint run ./...` (`make lint`).
- All Markdown files must pass `python3 scripts/validate_australian_english.py docs/ README.md` (`make validate-spelling`).
- Existing test contracts (`TestMask_BoundsAndOverlapErrors`, `TestPool_ZeroingOnRelease`, `TestCoalescer_SemaphoreFull`, `TestProcessor_LogBodyTypes`) must remain green.

## Review Focus

- Adjacent PII spans separated by JSON escaped quotes (`\"Alice\"\"bob@example.com\"`) where `widenJSONEscapes` causes widened boundaries to meet or overlap.
- Windows file paths containing `\u` directory prefixes (`C:\u1\alice` or `C:\\u0041\\alice`) that are not valid JSON `\uXXXX` escapes.
- Models or callers passing `numClasses > len(idToBIO)` to `DecodeWindow` or `len(a.Logits) == 0` (with `cap(a.Logits) > 0`) to `DetectBatch`.
- Attribute maps whose keys or values contain delimiter characters (`=` or `;`) such as `{"a": "b;c=d"}` vs `{"a": "b", "c": "d"}`.
- Oversized `BatchArena` slice growth during traffic bursts followed by `Pool.Release(a)` returning the arena to `sync.Pool`.

---

### Task 1: Span Masking Post-Widening Overlap Clamping & Hex-Validated `\uXXXX` Escapes (`internal/mask`)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/mask/mask.go:69-108,162-211`
- Modify: `processor/onnxredactionprocessor/internal/mask/mask_test.go`

**Interfaces:**
- Consumes: `types.EntitySpan`, `types.ErrorMode`
- Produces: `mask.Mask(text string, spans []types.EntitySpan, maxPayloadBytes int, mode types.ErrorMode, scratch []byte) (string, error)` (unchanged signature)

- [ ] **Step 1: Write failing tests in `processor/onnxredactionprocessor/internal/mask/mask_test.go`**

Add `TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths` to `processor/onnxredactionprocessor/internal/mask/mask_test.go`:

```go
func TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths(t *testing.T) {
	// 1. Two non-overlapping spans separated by escaped quotes \" where widening overlaps
	jsonAdj := `{"pair": "\"Alice\"\"bob@example.com\""}`
	sAlice := spanOf(jsonAdj, "Alice", types.LabelPerson)
	sBob := spanOf(jsonAdj, "bob@example.com", types.LabelEmail)
	outAdj, err := Mask(jsonAdj, []types.EntitySpan{sAlice, sBob}, 16384, types.ErrorModeDrop, nil)
	require.NoError(t, err, "post-widening overlap between non-overlapping raw spans must be clamped, not rejected")
	assert.Contains(t, outAdj, "[REDACTED_PERSON]")
	assert.Contains(t, outAdj, "[REDACTED_EMAIL]")
	assert.True(t, json.Valid([]byte(outAdj)), "masked output must remain valid JSON: %s", outAdj)

	// 2. Windows path with \u directory segment that is not hex \uXXXX
	winPath := `C:\u1\alice\secret.txt`
	sUser := spanOf(winPath, "alice", types.LabelPathUser)
	outWin, err := Mask(winPath, []types.EntitySpan{sUser}, 16384, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.Equal(t, `C:\u1\[REDACTED_PATH_USER]\secret.txt`, outWin)

	// 3. Escaped backslash before u0041 (\\u0041 is literal \ followed by u0041, not a unicode escape)
	escapedBackslash := `C:\\u0041\alice`
	sEsc := spanOf(escapedBackslash, "u0041", types.LabelPathUser)
	// Start at '0' (inside u0041) to verify \\u0041 is not widened across the escaped backslash
	sEsc.Start++
	outEsc, err := Mask(escapedBackslash, []types.EntitySpan{sEsc}, 16384, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(outEsc, `C:\\u`), "escaped backslash must not be treated as \\uXXXX: %s", outEsc)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths ./internal/mask`
Expected: FAIL (`mask: overlapping spans detected` on `jsonAdj` and corrupted `C:[REDACTED_PATH_USER]\secret.txt` on `winPath`).

- [ ] **Step 3: Implement raw overlap check, post-widening clamping, and `isValidUnicodeEscape` in `processor/onnxredactionprocessor/internal/mask/mask.go`**

1. In `Mask` (`mask.go:69-108`), track `rawLastEnd := 0` and `lastEnd := 0`:
   - Before `if s.Start >= inspectLimit`, check `if s.Start < rawLastEnd { return "", fmt.Errorf("mask: overlapping spans detected at offset %d", s.Start) }` and set `rawLastEnd = s.End`.
   - After `widenJSONEscapes` and `if end > inspectLimit`, clamp `if start < lastEnd { start = lastEnd }` and `if start >= end { continue }`.
2. In `widenJSONEscapes` (`mask.go:162-211`), add `isHexDigit(b byte) bool` and `isValidUnicodeEscape(text string, idx int) bool` (verifying `idx >= 0 && idx+6 <= len(text) && text[idx] == '\\' && text[idx+1] == 'u'`, an odd count of consecutive `\` ending at `idx`, and 4 hex digits at `text[idx+2:idx+6]`), and use `isValidUnicodeEscape(text, idx)` in steps 1 and 3 of `widenJSONEscapes`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd processor/onnxredactionprocessor && go test -race -v ./internal/mask/...`
Expected: PASS (all tests including `TestMask_BoundsAndOverlapErrors` and `TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths`).

- [ ] **Step 5: Commit**

```bash
git add processor/onnxredactionprocessor/internal/mask/mask.go processor/onnxredactionprocessor/internal/mask/mask_test.go
git commit -m "fix(mask): clamp post-widening span overlaps and validate hex in unicode escapes"
```

---

### Task 2: CGo Output Tensor Element Type Validation, Slice Reslicing & Decoder/Tokeniser Bounds (`internal/ort`, `internal/decode`, `internal/tokenizer`)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/ort/ort_shim.c:43-56,160-317`
- Modify: `processor/onnxredactionprocessor/internal/ort/engine.go:11,184-286`
- Modify: `processor/onnxredactionprocessor/internal/ort/engine_test.go`
- Modify: `processor/onnxredactionprocessor/internal/decode/decode.go:106-135`
- Modify: `processor/onnxredactionprocessor/internal/decode/decode_test.go`
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:319-379`
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`

**Interfaces:**
- Consumes: `arena.BatchArena`, `types.BucketedBatch`, `types.Offset`
- Produces: Hardened `ort.Engine.DetectBatch`, `decode.Decoder.DecodeWindow`, and `wordPieceTokenizer.tokenizeWordSlow` with unchanged external signatures.

- [ ] **Step 1: Write failing tests in `decode_test.go`, `engine_test.go`, and `tokenizer_test.go`**

1. In `processor/onnxredactionprocessor/internal/decode/decode_test.go`, add `TestDecodeWindow_MoreClassesThanConfigured`:
```go
func TestDecodeWindow_MoreClassesThanConfigured(t *testing.T) {
	// Decoder configured with only 3 classes ("O", "B-PERSON", "I-PERSON"),
	// but model outputs 25 classes with class 24 having highest logit.
	d := NewDecoder([]string{"O", "B-PERSON", "I-PERSON"}, nil, 0.85)
	numClasses := 25
	logits := make([]float32, numClasses)
	logits[24] = 20.0 // out of range of d.classToLabel (len 3)
	offsets := []types.Offset{{Start: 0, End: 5}}

	require.NotPanics(t, func() {
		spans := d.DecodeWindow(logits, numClasses, offsets, nil)
		assert.Empty(t, spans)
	})
}
```
2. In `processor/onnxredactionprocessor/internal/ort/engine_test.go`, add `TestEngine_ZeroLengthLogitsSliceAndZeroMaxSeqLen`:
```go
func TestEngine_ZeroLengthLogitsSliceAndZeroMaxSeqLen(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 1, false, dec)
	require.NoError(t, err)
	defer func() { _ = eng.Close() }()

	pool := arena.NewPool(4, 16, 19)
	a := pool.Acquire()
	defer pool.Release(a)

	// 1. MaxSeqLen == 0 must return nil, nil without panicking on &a.InputIDs[0]
	emptySeqBatch := types.BucketedBatch{
		MaxSeqLen: 0,
		Windows:   []types.TokenWindow{{ParentUniqueIdx: 0}},
	}
	spans, err := eng.DetectBatch(context.Background(), emptySeqBatch, a)
	require.NoError(t, err)
	assert.Nil(t, spans)

	// 2. Arena with len(a.Logits) == 0 but cap(a.Logits) > 0 must reslice and succeed
	a.Logits = a.Logits[:0]
	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 102},
		Offsets:         []types.Offset{{Start: 0, End: 0}, {Start: 0, End: 4}, {Start: 4, End: 4}},
	}
	validBatch := types.BucketedBatch{MaxSeqLen: 3, Windows: []types.TokenWindow{w}}
	spans, err = eng.DetectBatch(context.Background(), validBatch, a)
	require.NoError(t, err)
	require.Len(t, spans, 1)
}
```
3. In `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`, add `TestWordPieceTokenizer_SlowPathMultiByteRuneLenChange`:
```go
func TestWordPieceTokenizer_SlowPathMultiByteRuneLenChange(t *testing.T) {
	tok := &wordPieceTokenizer{
		vocab: map[string]int64{
			"[UNK]": 100, "[CLS]": 101, "[SEP]": 102,
			"i": 300, "##a": 301,
		},
		unkID:        100,
		clsID:        101,
		sepID:        102,
		prefix:       "##",
		maxInputChar: 512,
	}
	// 'İ' (U+0130) is 2 bytes in UTF-8, lowercases to 'i' (1 byte).
	// Repeat 'a' 255 times so prefixLen + rawLen > 256 (triggers tokenizeWordSlow).
	word := "İ" + strings.Repeat("a", 255)
	text := "prefix " + word + " suffix"
	ids, offsets, err := tok.Encode(text, nil, nil)
	require.NoError(t, err)
	require.NotEmpty(t, ids)
	for _, off := range offsets {
		assert.True(t, utf8.ValidString(text[off.Start:off.End]), "offset [%d:%d] must fall on valid UTF-8 rune boundaries", off.Start, off.End)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestDecodeWindow_MoreClassesThanConfigured|TestEngine_ZeroLengthLogitsSliceAndZeroMaxSeqLen|TestWordPieceTokenizer_SlowPathMultiByteRuneLenChange" ./internal/decode ./internal/ort ./internal/tokenizer`
Expected: FAIL (panics in `decode.go:106` and `engine.go:264/266`, and invalid UTF-8 offset in `tokenizer.go:366`).

- [ ] **Step 3: Implement fixes in `ort_shim.c`, `engine.go`, `decode.go`, and `tokenizer.go`**

1. In `processor/onnxredactionprocessor/internal/ort/ort_shim.c`:
   - In `ort_shim_create_session`, add `(void)concurrent_runs;`.
   - In `ort_shim_run_batch`, reject `batch_size <= 0 || seq_len <= 0` at entry, and after `GetTensorTypeAndShape`, call `g_ort->GetTensorElementType(shape_info, &elem_type)` and verify `elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`.
2. In `processor/onnxredactionprocessor/internal/ort/engine.go`:
   - Import `"runtime"`.
   - At the start of `DetectBatch`, return `nil, nil` if `len(batch.Windows) == 0 || batch.MaxSeqLen <= 0`.
   - Check `cap(a.InputIDs) < totalElements || cap(a.AttentionMask) < totalElements || cap(a.Logits) == 0`.
   - Reslice `a.Logits = a.Logits[:cap(a.Logits)]` before taking `logitsPtr := (*C.float)(unsafe.Pointer(&a.Logits[0]))`.
   - Call `runtime.KeepAlive(a)` immediately after `C.ort_shim_run_batch(...)`.
3. In `processor/onnxredactionprocessor/internal/decode/decode.go`:
   - In `DecodeWindow`, guard `bestClass < len(d.classToLabel)` before indexing `d.classToLabel[bestClass]`, `d.isB[bestClass]`, and `d.isI[bestClass]`.
4. In `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go`:
   - In `tokenizeWordSlow`, build lowercased `word` rune-by-rune while recording `textOffsets []int` mapping each byte in `word` to its relative start byte in `text[wordStart:wordEnd]`, and use `wordStart + textOffsets[subStart]` / `wordStart + textOffsets[subEnd]` when appending `dstOffsets`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd processor/onnxredactionprocessor && go test -race -v ./internal/decode/... ./internal/ort/... ./internal/tokenizer/...`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add processor/onnxredactionprocessor/internal/ort/ort_shim.c processor/onnxredactionprocessor/internal/ort/engine.go processor/onnxredactionprocessor/internal/ort/engine_test.go processor/onnxredactionprocessor/internal/decode/decode.go processor/onnxredactionprocessor/internal/decode/decode_test.go processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go
git commit -m "fix(ort,decode,tokenizer): validate float32 tensor type, slice bounds, and slow-path rune offsets"
```

---

### Task 3: Arena Retention Caps, Coalescer In-Place Span Merging, Processor Init Cleanup & Metrics Collision Bounds (`internal/arena`, `internal/coalesce`, `internal/metricsmerge`, `processor.go`, `README.md`)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/arena/arena.go`
- Modify: `processor/onnxredactionprocessor/internal/arena/arena_test.go`
- Modify: `processor/onnxredactionprocessor/internal/coalesce/coalesce.go:327,529-547`
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge.go:33-52,164-182,235-237`
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go`
- Modify: `processor/onnxredactionprocessor/processor.go:43-179`
- Modify: `processor/onnxredactionprocessor/processor_test.go`
- Modify: `README.md:269`

**Interfaces:**
- Consumes: `arena.Pool`, `coalesce.Coalescer`, `metricsmerge.ResolveCollisions`, `tokenizer.Tokenizer`
- Produces: Bounded arena buffers, zero-extra-slice span merging in `Coalescer`, leak-free `newProcessor` error cleanup, and explicit-bounds-safe `ResolveCollisions`.

- [ ] **Step 1: Write failing tests in `arena_test.go` and `merge_test.go`**

1. In `processor/onnxredactionprocessor/internal/arena/arena_test.go`, add `TestPool_OversizedSliceCapReset`:
```go
func TestPool_OversizedSliceCapReset(t *testing.T) {
	pool := NewPool(4, 16, 19) // tensorElements = 64
	a := pool.Acquire()

	// Grow TokenIDPool, OffsetPool, Windows, SpanScratch, and ScratchBytes past 4x baseline
	a.TokenIDPool = make([]int64, 1000)
	a.OffsetPool = make([]types.Offset, 1000)
	a.Windows = make([]types.TokenWindow, 100)
	a.SpanScratch = make([]types.EntitySpan, 500)
	a.ScratchBytes = make([]byte, 131072)

	// Slice InputIDs/Logits down as DetectBatch does
	a.InputIDs = a.InputIDs[:8]
	a.AttentionMask = a.AttentionMask[:8]
	a.Logits = a.Logits[:8*19]

	pool.Release(a)

	b := pool.Acquire()
	defer pool.Release(b)

	assert.Equal(t, 64, len(b.InputIDs), "InputIDs length must be restored to tensorElements")
	assert.Equal(t, 64*19, len(b.Logits), "Logits length must be restored to logitsElements")
	assert.LessOrEqual(t, cap(b.TokenIDPool), 256, "oversized TokenIDPool must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.OffsetPool), 256, "oversized OffsetPool must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.Windows), 16, "oversized Windows must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.SpanScratch), 256, "oversized SpanScratch must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.ScratchBytes), 65536, "oversized ScratchBytes must be reallocated to baseline")
}
```
2. In `processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go`, add `TestComputeCanonicalAttrHash_DelimiterCollisionAndHistogramBounds`:
```go
func TestComputeCanonicalAttrHash_DelimiterCollisionAndHistogramBounds(t *testing.T) {
	// 1. Delimiter collision resistance: {"a": "b;c=d"} vs {"a": "b", "c": "d"}
	m1 := pcommon.NewMap()
	m1.PutStr("a", "b;c=d")
	m2 := pcommon.NewMap()
	m2.PutStr("a", "b")
	m2.PutStr("c", "d")

	var scratch []string
	h1, scratch := ComputeCanonicalAttrHash(m1, scratch)
	h2, _ := ComputeCanonicalAttrHash(m2, scratch)
	assert.NotEqual(t, h1, h2, "maps with embedded '=' and ';' delimiters must not collide")

	// 2. Histogram with mismatched ExplicitBounds must not sum BucketCounts
	metric := pmetric.NewMetric()
	hist := metric.SetEmptyHistogram()
	ts := pcommon.Timestamp(1000)

	dp1 := hist.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("k", "v")
	dp1.SetTimestamp(ts)
	dp1.ExplicitBounds().FromRaw([]float64{10, 100})
	dp1.BucketCounts().FromRaw([]uint64{1, 2, 3})
	// dp1 has no Sum set

	dp2 := hist.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("k", "v")
	dp2.SetTimestamp(ts)
	dp2.ExplicitBounds().FromRaw([]float64{50, 500}) // different bounds!
	dp2.BucketCounts().FromRaw([]uint64{4, 5, 6})
	dp2.SetSum(42.0)

	ResolveCollisions(metric)
	require.Equal(t, 1, hist.DataPoints().Len())
	merged := hist.DataPoints().At(0)
	assert.Equal(t, []uint64{1, 2, 3}, merged.BucketCounts().AsRaw(), "mismatched explicit bounds must not sum bucket counts")
	assert.True(t, merged.HasSum())
	assert.Equal(t, 42.0, merged.Sum(), "Sum from second datapoint must be preserved when first datapoint has no Sum")
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestPool_OversizedSliceCapReset|TestComputeCanonicalAttrHash_DelimiterCollisionAndHistogramBounds" ./internal/arena ./internal/metricsmerge`
Expected: FAIL.

- [ ] **Step 3: Implement fixes in `arena.go`, `coalesce.go`, `merge.go`, `processor.go`, and `README.md`**

1. In `processor/onnxredactionprocessor/internal/arena/arena.go`:
   - Store `maxBatchSize`, `tensorElements`, and `logitsElements` on `Pool`.
   - In `Release(a)`:
     - Call `clear(a.InputIDs)` and restore `a.InputIDs = a.InputIDs[:cap(a.InputIDs)]`.
     - Call `clear(a.AttentionMask)` and restore `a.AttentionMask = a.AttentionMask[:cap(a.AttentionMask)]`.
     - Call `clear(a.Logits)` and restore `a.Logits = a.Logits[:cap(a.Logits)]`.
     - Call `clear(a.TokenIDPool)` (active length only); if `cap(a.TokenIDPool) > 4*p.tensorElements`, reallocate `make([]int64, 0, p.tensorElements)`, else reslice `[:0]`.
     - If `cap(a.OffsetPool) > 4*p.tensorElements`, reallocate `make([]types.Offset, 0, p.tensorElements)`, else reslice `[:0]`.
     - If `cap(a.Windows) > 4*p.maxBatchSize`, reallocate `make([]types.TokenWindow, 0, p.maxBatchSize)`, else reslice `[:0]`.
     - If `cap(a.SpanScratch) > 256`, reallocate `make([]types.EntitySpan, 0, 64)`, else reslice `[:0]`.
     - If `cap(a.ScratchBytes) > 65536`, clear and reallocate `make([]byte, 0, 16384)`, else `clear(a.ScratchBytes[:cap(a.ScratchBytes)])` and reslice `[:0]`.
2. In `processor/onnxredactionprocessor/internal/coalesce/coalesce.go`:
   - At line 327, preallocate `activeRequests := make([]*callerRequest, 0, len(batch))`.
   - At lines 529–547, merge in-place (`parentSpans[idx] = decode.MergeSpans(parentSpans[idx])`) and read `spans := parentSpans[idx]`, removing `mergedSpans`.
3. In `processor/onnxredactionprocessor/internal/metricsmerge/merge.go`:
   - Import `"slices"`.
   - In `ComputeCanonicalAttrHash`, hash `uint64(len(k))` before key bytes and `uint64(len(s))` before value bytes.
   - In `ResolveCollisions` (`Histogram` and `ExponentialHistogram`), add `else if !target.HasSum() && dp.HasSum() { target.SetSum(dp.Sum()) }`.
   - In `ResolveCollisions` (`Histogram`), check `target.BucketCounts().Len() == dp.BucketCounts().Len() && slices.Equal(target.ExplicitBounds().AsRaw(), dp.ExplicitBounds().AsRaw())`.
4. In `processor/onnxredactionprocessor/processor.go`:
   - Add `tokenizer tokenizer.Tokenizer` to `redactionProcessor`.
   - In `newProcessor`, if `tok` or `det` initialisation fails after `newTelemetry`, call `_ = tel.Close()` (and `_ = tok.Close()` if `det` fails).
   - In `Shutdown`, close `p.tokenizer` if non-nil.
5. In `README.md:269`:
   - Update `(`filter`, `bytes_body`)` to `(`filter`, `bytes_body`, `max_depth`)`.

- [ ] **Step 4: Run full test suite, benchmarks, and linters**

Run: `make fmt && make lint && make test && make bench`
Expected:
- `PASSED: All 14 markdown file(s) adhere to Australian English spelling.`
- `0 issues.`
- All 14 Go packages pass `-race` tests and benchmarks.

- [ ] **Step 5: Commit and push to `origin/fix/gatekeeper-remediation`**

```bash
git add processor/onnxredactionprocessor/internal/arena/arena.go processor/onnxredactionprocessor/internal/arena/arena_test.go processor/onnxredactionprocessor/internal/coalesce/coalesce.go processor/onnxredactionprocessor/internal/metricsmerge/merge.go processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go processor/onnxredactionprocessor/processor.go README.md docs/superpowers/specs/2026-10-24-gatekeeper-round2-design.md docs/superpowers/plans/2026-10-24-gatekeeper-round2.md
git commit -m "fix(arena,coalesce,metricsmerge,processor): bound arena retention, harden metric collisions, and clean up init errors"
git push origin fix/gatekeeper-remediation
```
