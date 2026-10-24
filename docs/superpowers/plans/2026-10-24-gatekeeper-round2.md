# Gatekeeper Review Round 2 Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:dispatching-parallel-agents and superpowers:subagent-driven-development to implement Tasks 1–4 concurrently across strictly disjoint packages, followed by superpowers:test-driven-development (Red-Green-Refactor) within each subagent. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all validated P0, P1, P2, and documentation findings from the Round 2 static analysis and architectural review using test-driven development and parallel subagents.

**Architecture:** Partition the 8 validated findings into 4 strictly disjoint, zero-shared-file parallel subagent tracks with unchanged cross-package signatures:
- **Track A (Task 1 — `internal/mask`):** Post-widening span overlap clamping and hex-validated `\uXXXX` JSON escapes.
- **Track B (Task 2 — `internal/ort`, `internal/decode`):** CGo `FLOAT` output tensor element type validation, overflow-safe capacity bounds, slice reslicing, `runtime.KeepAlive`, and decoder class bounds checking.
- **Track C (Task 3 — `internal/tokenizer`, `internal/metricsmerge`):** Slow-path multi-byte UTF-8 rune offset mapping, type/length-prefixed canonical attribute hashing, and explicit histogram bounds validation.
- **Track D (Task 4 — `internal/arena`, `internal/coalesce`, `processor.go`, `README.md`):** Multi-sub-batch safe arena zeroing with `4x` retention caps, coalescer in-place span merging, deferred `newProcessor` error rollback / tokeniser shutdown, and `README.md` metric documentation.

**Tech Stack:** Go 1.27, CGo (`onnxruntime_c_api.h` v1.31.0), OpenTelemetry Collector `pdata` v0.163.0, `stretchr/testify`.

**Spec:** `docs/superpowers/specs/2026-10-24-gatekeeper-round2-design.md`

## Global Constraints

- All code must compile and pass `go test -race -count=1 ./...` (`make test`) and `golangci-lint run ./...` (`make lint`).
- All Markdown files must pass `python3 scripts/validate_australian_english.py docs/ README.md` (`make validate-spelling`).
- Existing test contracts (`TestMask_BoundsAndOverlapErrors`, `TestPool_ZeroingOnRelease`, `TestCoalescer_SemaphoreFull`, `TestProcessor_LogBodyTypes`) must remain green.
- Parallel subagents must modify **only** the files listed in their assigned task and must **not** run `git commit` concurrently (preventing `.git/index.lock` races); the coordinator commits each track atomically after parallel completion and review.

## Parallel Subagent Topology & File Isolation Matrix

| Parallel Track | Task | Owned Packages | Owned Files | Cross-Track File Overlap | Signature Drift |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Track A** | Task 1 | `./internal/mask` | `internal/mask/mask.go`, `internal/mask/mask_test.go` | None | None |
| **Track B** | Task 2 | `./internal/ort`, `./internal/decode` | `internal/ort/ort_shim.c`, `internal/ort/engine.go`, `internal/ort/engine_test.go`, `internal/decode/decode.go`, `internal/decode/decode_test.go` | None | None |
| **Track C** | Task 3 | `./internal/tokenizer`, `./internal/metricsmerge` | `internal/tokenizer/tokenizer.go`, `internal/tokenizer/tokenizer_test.go`, `internal/metricsmerge/merge.go`, `internal/metricsmerge/merge_test.go` | None | None |
| **Track D** | Task 4 | `./internal/arena`, `./internal/coalesce`, root `onnxredactionprocessor` | `internal/arena/arena.go`, `internal/arena/arena_test.go`, `internal/coalesce/coalesce.go`, `processor.go`, `processor_test.go`, `README.md` | None | None |

## Review Focus

- Non-overlapping raw spans that touch inside a `\uXXXX` or `\"` JSON escape so `widenJSONEscapes` expands both spans into overlapping byte ranges.
- Windows file paths containing `\u` directory prefixes (`C:\u1\alice` or `C:\\u0041\alice`) that are not valid JSON `\uXXXX` escapes.
- Models or callers passing `numClasses > len(idToBIO)` to `DecodeWindow` or `len(a.Logits) == 0` (with `cap(a.Logits) > 0`) to `DetectBatch`.
- Attribute maps whose keys or values contain delimiter characters (`=` or `;`) such as `{"a": "b;c=d"}` vs `{"a": "b", "c": "d"}`, or string vs integer values (`{"a": "1"}` vs `{"a": 1}`).
- Multi-sub-batch `DetectBatch` calls where Sub-batch 1 dirties a larger slice prefix than Sub-batch 2 before `Pool.Release(a)` zeroes the arena.

---

### Task 1 (Parallel Track A): Span Masking Post-Widening Overlap Clamping & Hex-Validated `\uXXXX` Escapes (`internal/mask`)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/mask/mask.go:69-108,162-211`
- Modify: `processor/onnxredactionprocessor/internal/mask/mask_test.go`

**Interfaces:**
- Consumes: `types.EntitySpan`, `types.ErrorMode`
- Produces: `mask.Mask(text string, spans []types.EntitySpan, maxPayloadBytes int, mode types.ErrorMode, scratch []byte) (string, error)` (unchanged signature)

- [x] **Step 1: Write the failing test in `processor/onnxredactionprocessor/internal/mask/mask_test.go`**

Add `TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths` to `processor/onnxredactionprocessor/internal/mask/mask_test.go`:

```go
func TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths(t *testing.T) {
	// 1. Two non-overlapping raw spans meeting inside \u0020 (or \") where widening overlaps:
	// Span 1 ends at '0' inside \u0020 (uIdx+3 -> Rule 3 widens end to uIdx+6),
	// Span 2 starts at '0' inside \u0020 (uIdx+3 -> Rule 1 widens start to uIdx).
	jsonAdj := `{"pair": "Alice\u0020bob@example.com"}`
	uIdx := strings.Index(jsonAdj, `\u0020`)
	sAlice := types.EntitySpan{
		Start:      strings.Index(jsonAdj, "Alice"),
		End:        uIdx + 3,
		Label:      types.LabelPerson,
		Confidence: 0.95,
	}
	sBob := types.EntitySpan{
		Start:      uIdx + 3,
		End:        strings.Index(jsonAdj, "bob@example.com") + len("bob@example.com"),
		Label:      types.LabelEmail,
		Confidence: 0.95,
	}
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
	sEsc.Start++ // Start at '0' (inside u0041) to verify \\u0041 is not widened across the escaped backslash
	outEsc, err := Mask(escapedBackslash, []types.EntitySpan{sEsc}, 16384, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(outEsc, `C:\\u`), "escaped backslash must not be treated as \\uXXXX: %s", outEsc)
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd processor/onnxredactionprocessor && go test -v -run TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths ./internal/mask`
Expected: FAIL (`mask: overlapping spans detected at offset 15` on `jsonAdj` and `C:[REDACTED_PATH_USER]\secret.txt` on `winPath`).

- [x] **Step 3: Implement raw overlap check, post-widening clamping, and `isValidUnicodeEscape` in `processor/onnxredactionprocessor/internal/mask/mask.go`**

1. In `Mask` (`mask.go:69-108`), track `rawLastEnd := 0` and `lastEnd := 0`:
   - Before `if s.Start >= inspectLimit`, check `if s.Start < rawLastEnd { return "", fmt.Errorf("mask: overlapping spans detected at offset %d", s.Start) }` and set `rawLastEnd = s.End`.
   - After `widenJSONEscapes` and `if end > inspectLimit`, clamp `if start < lastEnd { start = lastEnd }` and skip empty spans (`if start >= end { continue }`).
2. In `widenJSONEscapes` (`mask.go:162-211`), add `isHexDigit(b byte) bool` and `isValidUnicodeEscape(text string, idx int) bool` (verifying `idx >= 0 && idx+6 <= len(text) && text[idx] == '\\' && text[idx+1] == 'u'`, an odd count of consecutive `\` ending at `idx`, and 4 hex digits at `text[idx+2:idx+6]`), and use `isValidUnicodeEscape(text, idx)` in steps 1 and 3 of `widenJSONEscapes`.

- [x] **Step 4: Run package tests to verify they pass**

Run: `cd processor/onnxredactionprocessor && go test -race -count=1 -v ./internal/mask/...`
Expected: PASS (including `TestMask_BoundsAndOverlapErrors` and `TestMask_AdjacentWidenedJSONEscapesAndWindowsPaths`).

- [x] **Step 5: Stage and commit Track A changes (Coordinator)**

```bash
git add processor/onnxredactionprocessor/internal/mask/mask.go processor/onnxredactionprocessor/internal/mask/mask_test.go
git commit -m "fix(mask): clamp post-widening span overlaps and validate hex in unicode escapes"
```

---

### Task 2 (Parallel Track B): CGo Output Tensor Element Type Validation, Slice Reslicing & Decoder Class Bounds (`internal/ort`, `internal/decode`)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/ort/ort_shim.c:43-56,160-317`
- Modify: `processor/onnxredactionprocessor/internal/ort/engine.go:11,136,184-286`
- Modify: `processor/onnxredactionprocessor/internal/ort/engine_test.go`
- Modify: `processor/onnxredactionprocessor/internal/decode/decode.go:106-135`
- Modify: `processor/onnxredactionprocessor/internal/decode/decode_test.go`

**Interfaces:**
- Consumes: `arena.BatchArena`, `types.BucketedBatch`, `types.Offset`
- Produces: `(*ort.Engine).DetectBatch` and `(*decode.Decoder).DecodeWindow` with unchanged external signatures.

- [x] **Step 1: Write failing tests in `decode_test.go` and `engine_test.go`**

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

- [x] **Step 2: Run tests to verify they fail**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestDecodeWindow_MoreClassesThanConfigured|TestEngine_ZeroLengthLogitsSliceAndZeroMaxSeqLen" ./internal/decode ./internal/ort`
Expected: FAIL (panics in `decode.go:106` and `engine.go:264/266`).

- [x] **Step 3: Implement fixes in `ort_shim.c`, `engine.go`, and `decode.go`**

1. In `processor/onnxredactionprocessor/internal/ort/ort_shim.c`:
   - In `ort_shim_create_session`, add `(void)concurrent_runs;`.
   - In `ort_shim_run_batch`, reject `batch_size <= 0 || seq_len <= 0 || out_logits_cap <= 0` at entry, verify `elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT` via `g_ort->GetTensorElementType(shape_info, &elem_type)`, and check `if (num_classes > out_logits_cap / (batch_size * seq_len))` before multiplying `total_elements = batch_size * seq_len * num_classes`.
2. In `processor/onnxredactionprocessor/internal/ort/engine.go`:
   - Import `"runtime"`.
   - In `NewEngine` (`engine.go:136`), size `warmArena := arena.NewPool(1, 2, 64).Acquire()`.
   - At the start of `DetectBatch`, return `nil, nil` if `len(batch.Windows) == 0 || batch.MaxSeqLen <= 0`.
   - Check `cap(a.InputIDs) < totalElements || cap(a.AttentionMask) < totalElements || cap(a.Logits) == 0`.
   - Reslice `a.Logits = a.Logits[:cap(a.Logits)]` before taking `logitsPtr := (*C.float)(unsafe.Pointer(&a.Logits[0]))`.
   - Call `runtime.KeepAlive(a)` immediately after `C.ort_shim_run_batch(...)`.
3. In `processor/onnxredactionprocessor/internal/decode/decode.go`:
   - In `DecodeWindow`, guard `bestClass < len(d.classToLabel)` before indexing `d.classToLabel[bestClass]`.

- [x] **Step 4: Run package tests to verify they pass**

Run: `cd processor/onnxredactionprocessor && go test -race -count=1 -v ./internal/decode/... ./internal/ort/...`
Expected: PASS.

- [x] **Step 5: Stage and commit Track B changes (Coordinator)**

```bash
git add processor/onnxredactionprocessor/internal/ort/ort_shim.c processor/onnxredactionprocessor/internal/ort/engine.go processor/onnxredactionprocessor/internal/ort/engine_test.go processor/onnxredactionprocessor/internal/decode/decode.go processor/onnxredactionprocessor/internal/decode/decode_test.go
git commit -m "fix(ort,decode): validate float32 tensor element type, slice bounds, and decoder class index"
```

---

### Task 3 (Parallel Track C): Slow-Path Multi-Byte Rune Offset Mapping & Metric Collision Bounds (`internal/tokenizer`, `internal/metricsmerge`)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go:319-379`
- Modify: `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge.go:1-54,164-182,235-237`
- Modify: `processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go`

**Interfaces:**
- Consumes: `types.Offset`, `pcommon.Map`, `pmetric.Metric`
- Produces: `(*wordPieceTokenizer).Encode`, `metricsmerge.ComputeCanonicalAttrHash`, `metricsmerge.ResolveCollisions` (unchanged signatures)

- [x] **Step 1: Write failing tests in `tokenizer_test.go` and `merge_test.go`**

1. In `processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go`, add `TestWordPieceTokenizer_SlowPathMultiByteRuneLenChange`:
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

	// 2. Type collision resistance: {"a": "123"} vs {"a": 123}
	mStr := pcommon.NewMap()
	mStr.PutStr("a", "123")
	mInt := pcommon.NewMap()
	mInt.PutInt("a", 123)
	hStr, scratch := ComputeCanonicalAttrHash(mStr, scratch)
	hInt, _ := ComputeCanonicalAttrHash(mInt, scratch)
	assert.NotEqual(t, hStr, hInt, "string attribute '123' and int attribute 123 must not collide")

	// 3. Histogram with mismatched ExplicitBounds must not sum BucketCounts
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

- [x] **Step 2: Run tests to verify they fail**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestWordPieceTokenizer_SlowPathMultiByteRuneLenChange|TestComputeCanonicalAttrHash_DelimiterCollisionAndHistogramBounds" ./internal/tokenizer ./internal/metricsmerge`
Expected: FAIL.

- [x] **Step 3: Implement fixes in `tokenizer.go` and `merge.go`**

1. In `processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go` (`tokenizeWordSlow`, lines 319–379):
   - Lowercase `text[wordStart:wordEnd]` rune-by-rune while recording `textOffsets []int` mapping each byte in lowercased `word` to its relative start byte in `text[wordStart:wordEnd]` (with `textOffsets[len(word)] = wordEnd - wordStart`), and use `wordStart + textOffsets[subStart]` and `wordStart + textOffsets[subEnd]` when appending `dstOffsets`.
2. In `processor/onnxredactionprocessor/internal/metricsmerge/merge.go`:
   - Import `"slices"`.
   - In `ComputeCanonicalAttrHash`, mix `uint64(len(k))` (`h ^= uint64(len(k)); h *= fnvPrime64`) before hashing key bytes, and mix `uint64(v.Type())` and `uint64(len(s))` before hashing value bytes.
   - In `ResolveCollisions` (`Histogram` at line 164 and `ExponentialHistogram` at line 235), add `else if !target.HasSum() && dp.HasSum() { target.SetSum(dp.Sum()) }`.
   - In `ResolveCollisions` (`Histogram` at line 178), check `target.BucketCounts().Len() == dp.BucketCounts().Len() && slices.Equal(target.ExplicitBounds().AsRaw(), dp.ExplicitBounds().AsRaw())`.

- [x] **Step 4: Run package tests to verify they pass**

Run: `cd processor/onnxredactionprocessor && go test -race -count=1 -v ./internal/tokenizer/... ./internal/metricsmerge/...`
Expected: PASS.

- [x] **Step 5: Stage and commit Track C changes (Coordinator)**

```bash
git add processor/onnxredactionprocessor/internal/tokenizer/tokenizer.go processor/onnxredactionprocessor/internal/tokenizer/tokenizer_test.go processor/onnxredactionprocessor/internal/metricsmerge/merge.go processor/onnxredactionprocessor/internal/metricsmerge/merge_test.go
git commit -m "fix(tokenizer,metricsmerge): map slow-path rune offsets and harden metric collision hashing and bounds"
```

---

### Task 4 (Parallel Track D): Arena Retention Caps, Coalescer In-Place Span Merging, Processor Init Cleanup & `README.md` (`internal/arena`, `internal/coalesce`, `processor.go`, `README.md`)

**Files:**
- Modify: `processor/onnxredactionprocessor/internal/arena/arena.go`
- Modify: `processor/onnxredactionprocessor/internal/arena/arena_test.go`
- Modify: `processor/onnxredactionprocessor/internal/coalesce/coalesce.go:327,529-547`
- Modify: `processor/onnxredactionprocessor/processor.go:43-179`
- Modify: `processor/onnxredactionprocessor/processor_test.go`
- Modify: `README.md:269`

**Interfaces:**
- Consumes: `arena.Pool`, `coalesce.Coalescer`, `tokenizer.Tokenizer`
- Produces: Bounded `arena.Pool.Release`, allocation-reduced `coalesce.Coalescer.executeBatch`, leak-free `newProcessor` error path and `Shutdown`.

- [x] **Step 1: Write failing tests in `arena_test.go` and `processor_test.go`**

1. In `processor/onnxredactionprocessor/internal/arena/arena_test.go`, add `TestPool_OversizedSliceCapReset`:
```go
func TestPool_OversizedSliceCapReset(t *testing.T) {
	pool := NewPool(4, 16, 19) // tensorElements = 64
	a := pool.Acquire()

	// Dirty index 30 (simulating Sub-batch 1), then slice down to [:8] (simulating Sub-batch 2)
	a.InputIDs[30] = 999
	a.AttentionMask[30] = 1
	a.Logits[30] = 42.0
	a.InputIDs = a.InputIDs[:8]
	a.AttentionMask = a.AttentionMask[:8]
	a.Logits = a.Logits[:8*19]

	// Grow TokenIDPool, OffsetPool, Windows, SpanScratch, and ScratchBytes past 4x baseline
	a.TokenIDPool = make([]int64, 1000)
	a.OffsetPool = make([]types.Offset, 1000)
	a.Windows = make([]types.TokenWindow, 100)
	a.SpanScratch = make([]types.EntitySpan, 500)
	a.ScratchBytes = make([]byte, 131072)

	pool.Release(a)

	b := pool.Acquire()
	defer pool.Release(b)

	assert.Equal(t, 64, len(b.InputIDs), "InputIDs length must be restored to tensorElements")
	assert.Equal(t, 64*19, len(b.Logits), "Logits length must be restored to logitsElements")
	assert.Equal(t, int64(0), b.InputIDs[30], "InputIDs beyond final sub-batch slice must still be zeroed")
	assert.Equal(t, int64(0), b.AttentionMask[30], "AttentionMask beyond final sub-batch slice must still be zeroed")
	assert.Equal(t, float32(0), b.Logits[30], "Logits beyond final sub-batch slice must still be zeroed")
	assert.LessOrEqual(t, cap(b.TokenIDPool), 256, "oversized TokenIDPool must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.OffsetPool), 256, "oversized OffsetPool must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.Windows), 16, "oversized Windows must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.SpanScratch), 256, "oversized SpanScratch must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.ScratchBytes), 65536, "oversized ScratchBytes must be reallocated to baseline")
}
```
2. In `processor/onnxredactionprocessor/processor_test.go`, add `TestProcessor_TokenizerCloseOnShutdownAndInitError`:
```go
type closeTrackingTokenizer struct {
	tokenizer.Tokenizer
	closed int
}

func (c *closeTrackingTokenizer) Close() error {
	c.closed++
	return nil
}

func TestProcessor_TokenizerCloseOnShutdownAndInitError(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Model.TokenizerPath = filepath.Clean("testdata/tokenizer.json")
	cfg.Model.Path = "testdata/nonexistent_model.onnx"

	baseTok, err := tokenizer.NewWordPieceTokenizer(cfg.Model.TokenizerPath, "")
	require.NoError(t, err)
	trackTok := &closeTrackingTokenizer{Tokenizer: baseTok}

	set := processortest.NewNopSettings(Type)
	// 1. When detector init fails, newProcessor must close tokenizer and telemetry
	_, err = newProcessor(cfg, &set.TelemetrySettings, trackTok, nil)
	require.Error(t, err)
	assert.Equal(t, 1, trackTok.closed, "tokenizer must be closed when newProcessor fails during engine init")

	// 2. When processor succeeds and Shutdown is called, tokenizer.Close() must be called
	trackTok2 := &closeTrackingTokenizer{Tokenizer: baseTok}
	proc, err := newProcessor(cfg, &set.TelemetrySettings, trackTok2, &stubDetector{})
	require.NoError(t, err)
	require.NoError(t, proc.Shutdown(context.Background()))
	assert.Equal(t, 1, trackTok2.closed, "tokenizer must be closed on processor Shutdown")
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd processor/onnxredactionprocessor && go test -v -run "TestPool_OversizedSliceCapReset|TestProcessor_TokenizerCloseOnShutdownAndInitError" ./internal/arena .`
Expected: FAIL.

- [x] **Step 3: Implement fixes in `arena.go`, `coalesce.go`, `processor.go`, and `README.md`**

1. In `processor/onnxredactionprocessor/internal/arena/arena.go`:
   - Store `maxBatchSize`, `tensorElements`, and `logitsElements` on `Pool`.
   - In `Release(a)`:
     - Reslice `a.InputIDs = a.InputIDs[:cap(a.InputIDs)]` and call `clear(a.InputIDs)`.
     - Reslice `a.AttentionMask = a.AttentionMask[:cap(a.AttentionMask)]` and call `clear(a.AttentionMask)`.
     - Reslice `a.Logits = a.Logits[:cap(a.Logits)]` and call `clear(a.Logits)`.
     - Call `clear(a.TokenIDPool)` (active length only); if `cap(a.TokenIDPool) > 4*p.tensorElements`, reallocate `make([]int64, 0, p.tensorElements)`, else reslice `[:0]`.
     - If `cap(a.OffsetPool) > 4*p.tensorElements`, reallocate `make([]types.Offset, 0, p.tensorElements)`, else reslice `[:0]`.
     - If `cap(a.Windows) > 4*p.maxBatchSize`, reallocate `make([]types.TokenWindow, 0, p.maxBatchSize)`, else reslice `[:0]`.
     - If `cap(a.SpanScratch) > 256`, reallocate `make([]types.EntitySpan, 0, 64)`, else reslice `[:0]`.
     - If `cap(a.ScratchBytes) > 65536`, clear and reallocate `make([]byte, 0, 16384)`, else `clear(a.ScratchBytes[:cap(a.ScratchBytes)])` and reslice `[:0]`.
2. In `processor/onnxredactionprocessor/internal/coalesce/coalesce.go`:
   - At line 327, preallocate `activeRequests := make([]*callerRequest, 0, len(batch))`.
   - At lines 529–547, merge spans in-place (`parentSpans[idx] = decode.MergeSpans(parentSpans[idx])`) and read `spans := parentSpans[idx]`, removing `mergedSpans`.
3. In `processor/onnxredactionprocessor/processor.go`:
   - Add `tokenizer tokenizer.Tokenizer` to `redactionProcessor`.
   - In `newProcessor`, register a `var success bool` with `defer func() { if !success { _ = tel.Close(); if tok != nil { _ = tok.Close() }; if det != nil { _ = det.Close() } } }()`, and set `success = true` immediately before returning `&redactionProcessor{...}, nil`.
   - Populate `tokenizer: tok` on `&redactionProcessor{...}` and close `p.tokenizer` in `Shutdown`.
4. In `README.md:269`:
   - Update `(`filter`, `bytes_body`)` to `(`filter`, `bytes_body`, `max_depth`)`.

- [x] **Step 4: Run package tests to verify they pass**

Run: `cd processor/onnxredactionprocessor && go test -race -count=1 -v ./internal/arena/... ./internal/coalesce/... .`
Expected: PASS.

- [x] **Step 5: Stage and commit Track D changes, run full verification, and push (Coordinator)**

```bash
git add processor/onnxredactionprocessor/internal/arena/arena.go processor/onnxredactionprocessor/internal/arena/arena_test.go processor/onnxredactionprocessor/internal/coalesce/coalesce.go processor/onnxredactionprocessor/processor.go processor/onnxredactionprocessor/processor_test.go README.md
git commit -m "fix(arena,coalesce,processor): cap arena retention, merge spans in-place, and clean up init/shutdown resources"
make fmt && make lint && make test && make bench
git push origin fix/gatekeeper-remediation
```
