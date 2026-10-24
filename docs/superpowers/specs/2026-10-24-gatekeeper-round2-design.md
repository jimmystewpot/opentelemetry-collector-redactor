# Pre-Production Gatekeeper Review (Round 2) Remediation Specification

**Date:** 2026-10-24
**Target Branch:** `fix/gatekeeper-remediation`
**Go Version:** 1.27

## 1. Objective

Remediate all validated P0, P1, P2, and documentation findings from the second-pass static analysis and memory safety audit while preserving existing test contracts, zero-allocation hot paths, and 100% Australian English documentation compliance.

## 2. Validated Findings & Architectural Decisions

1. **P0-1 (`internal/mask`): Post-Widening Span Overlap & Hex-Validated `\uXXXX` Escapes**
   - Preserve raw overlap rejection (`s.Start < rawLastEnd`) prior to calling `widenJSONEscapes` so `TestMask_BoundsAndOverlapErrors` remains intact.
   - After `widenJSONEscapes`, clamp `if start < lastEnd { start = lastEnd }` and skip empty spans (`start >= end`) so adjacent entities whose boundaries touch or split a JSON escape (`\"` or `\uXXXX`) are both masked cleanly without triggering false overlap errors.
   - Require `isValidUnicodeEscape(text, idx)` (odd backslash run length ending at `idx` and 4 hexadecimal characters at `idx+2..idx+6`) before widening across `\uXXXX`, preventing corruption of Windows paths such as `C:\u1\alice` or `C:\\u0041\alice`.
2. **P0-2 (`internal/ort`, `internal/decode`): CGo Output Tensor Element Type, Overflow-Safe Capacity Check, Slice Reslicing & Decoder Class Bounds**
   - In `ort_shim.c`, verify `batch_size > 0 && seq_len > 0 && out_logits_cap > 0`, verify `GetTensorElementType(shape_info, &elem_type) == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`, and guard against 64-bit multiplication overflow (`num_classes > out_logits_cap / (batch_size * seq_len)`) before `memcpy`.
   - In `engine.go`, return early if `len(batch.Windows) == 0 || batch.MaxSeqLen <= 0`, size `warmupArena` for up to 64 classes (`arena.NewPool(1, 2, 64)`), reslice `a.Logits = a.Logits[:cap(a.Logits)]` before taking `&a.Logits[0]`, and call `runtime.KeepAlive(a)` after `C.ort_shim_run_batch`.
   - In `decode.go`, guard `bestClass < len(d.classToLabel)` in `DecodeWindow` so models with `numClasses > len(idToBIO)` treat unmapped classes as `types.LabelNone` (`"O"`).
3. **P1-1 (`internal/arena`): Full-Capacity Security Zeroing Across Multi-Batch Runs & Oversized Buffer Retention Caps**
   - In `Release(a)`, reslice `InputIDs`, `AttentionMask`, and `Logits` to `[:cap]` **before** calling `clear(...)` so that if an earlier sub-batch dirtied a larger range than a subsequent sub-batch in the same `executeBatch` call, the entire dirtied range is guaranteed to be zeroed and `Acquire()` always receives `len == cap`.
   - For append-only `TokenIDPool`, call `clear(a.TokenIDPool)` on active `[:len]`, and reallocate `TokenIDPool`, `OffsetPool`, `Windows`, `SpanScratch`, or `ScratchBytes` back to baseline capacity if they grew past `4x` baseline during a traffic burst.
4. **P1-2 (`internal/coalesce`): In-Place Span Merging & Preallocated Request Slice**
   - Merge spans in-place into `parentSpans[idx]` instead of allocating `mergedSpans`, and preallocate `activeRequests`.
5. **P1-3 (`processor.go`): Deferred Init Rollback & Tokeniser Shutdown**
   - Use a `var success bool` guard with `defer` in `newProcessor` so any early error automatically closes `tel`, `tok`, and `det`, and store `tokenizer` on `redactionProcessor` to close it during `Shutdown`.
6. **P2-1 (`internal/metricsmerge`): Explicit Histogram Bounds Validation, `HasSum` Preservation & Type/Length-Prefixed Canonical Hash**
   - Mix `uint64(len(k))`, `uint64(v.Type())`, and `uint64(len(s))` into `ComputeCanonicalAttrHash`, verify `slices.Equal(target.ExplicitBounds().AsRaw(), dp.ExplicitBounds().AsRaw())` before summing explicit histogram buckets, and copy `dp.Sum()` when `!target.HasSum() && dp.HasSum()`.
7. **P2-2 (`internal/tokenizer`): Multi-Byte Rune Offset Mapping in `tokenizeWordSlow`**
   - Track original UTF-8 byte offsets per lowercased byte in `tokenizeWordSlow` when `unicode.ToLower` alters rune byte length.
8. **DOC-1 (`README.md`): Document `max_depth` Skip Reason**
   - Add `max_depth` to the `redaction_skipped_total` description in `README.md`.
