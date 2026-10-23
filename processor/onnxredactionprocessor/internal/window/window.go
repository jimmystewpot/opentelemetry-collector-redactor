package window

import (
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// Chunker slices oversized inputs at UTF-8 rune boundaries and partitions
// token streams into sliding windows with configurable overlap.
type Chunker struct {
	windowTokens    int
	overlapTokens   int
	step            int
	maxPayloadBytes int
}

// NewChunker constructs a Chunker with the given window size, overlap, and payload cap.
func NewChunker(windowTokens, overlapTokens, maxPayloadBytes int) *Chunker {
	if windowTokens <= 0 {
		windowTokens = 256
	}
	step := windowTokens - overlapTokens
	if step <= 0 {
		step = windowTokens
	}
	return &Chunker{
		windowTokens:    windowTokens,
		overlapTokens:   overlapTokens,
		step:            step,
		maxPayloadBytes: maxPayloadBytes,
	}
}

// SliceInput truncates text at the last UTF-8 rune boundary <= maxPayloadBytes
// and returns (inspectedPrefix, hasUninspectedTail).
func (c *Chunker) SliceInput(text string) (string, bool) {
	if c.maxPayloadBytes <= 0 || len(text) <= c.maxPayloadBytes {
		return text, false
	}

	cut := c.maxPayloadBytes
	for cut > 0 && (text[cut]&0xC0) == 0x80 {
		cut--
	}

	return text[:cut], true
}

// WindowTokens splits token IDs and offsets into windows of at most windowTokens
// with overlapTokens overlap (step = windowTokens - overlapTokens).
func (c *Chunker) WindowTokens(parentIdx int, ids []int64, offsets []types.Offset, dst []types.TokenWindow) []types.TokenWindow {
	n := len(ids)
	if n == 0 {
		return dst
	}

	if n <= c.windowTokens {
		return append(dst, types.TokenWindow{
			ParentUniqueIdx: parentIdx,
			TokenIDs:        ids,
			Offsets:         offsets,
		})
	}

	for start := 0; start < n; start += c.step {
		end := start + c.windowTokens
		if end > n {
			end = n
		}

		dst = append(dst, types.TokenWindow{
			ParentUniqueIdx: parentIdx,
			TokenIDs:        ids[start:end],
			Offsets:         offsets[start:end],
		})

		if end == n {
			break
		}
	}

	return dst
}
