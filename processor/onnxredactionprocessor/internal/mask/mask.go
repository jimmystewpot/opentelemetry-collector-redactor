package mask

import (
	"errors"
	"fmt"
	"sort"
	"unicode/utf8"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// Mask applies non-overlapping sorted spans back-to-front or left-to-right using [REDACTED_<LABEL>].
// If scratch is provided, writes into scratch to avoid intermediate allocations.
// If maxPayloadBytes > 0 and len(text) > maxPayloadBytes:
//   - under ErrorModeDrop, bytes at/after the last rune boundary <= maxPayloadBytes
//     are replaced with types.RedactedUninspected.
//   - under ErrorModeIgnore, the uninspected tail is preserved unchanged.
//
// Returns an error if text is invalid UTF-8 or any span violates bounds/rune boundaries.
func Mask(text string, spans []types.EntitySpan, maxPayloadBytes int, mode types.ErrorMode, scratch []byte) (string, error) {
	if !utf8.ValidString(text) {
		return "", errors.New("mask: invalid UTF-8 string")
	}

	textLen := len(text)
	isOversize := maxPayloadBytes > 0 && textLen > maxPayloadBytes

	if len(spans) == 0 && !isOversize {
		return text, nil
	}

	// Determine inspect boundary
	inspectLimit := textLen
	if isOversize {
		inspectLimit = maxPayloadBytes
		for inspectLimit > 0 && !isRuneBoundary(text, inspectLimit) {
			inspectLimit--
		}
	}

	// Validate, widen, and filter spans
	type validSpan struct {
		start       int
		end         int
		placeholder string
	}

	var processed []validSpan
	var stackProcessed [16]validSpan
	if len(spans) <= 16 {
		processed = stackProcessed[:0]
	} else {
		processed = make([]validSpan, 0, len(spans))
	}

	if len(spans) > 0 {
		var sorted []types.EntitySpan
		var stackSorted [16]types.EntitySpan
		if len(spans) <= 16 {
			sorted = stackSorted[:len(spans)]
		} else {
			sorted = make([]types.EntitySpan, len(spans))
		}
		copy(sorted, spans)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].Start < sorted[j].Start
		})

		rawLastEnd := 0
		lastEnd := 0
		for _, s := range sorted {
			if s.Start < 0 || s.End > textLen || s.Start > s.End {
				return "", fmt.Errorf("mask: span out of bounds [%d, %d] for text len %d", s.Start, s.End, textLen)
			}
			if !isRuneBoundary(text, s.Start) || !isRuneBoundary(text, s.End) {
				return "", fmt.Errorf("mask: span [%d, %d] does not fall on UTF-8 rune boundaries", s.Start, s.End)
			}
			if s.Start < rawLastEnd {
				return "", fmt.Errorf("mask: overlapping spans detected at offset %d", s.Start)
			}
			rawLastEnd = s.End

			// Drop spans entirely after inspectLimit
			if s.Start >= inspectLimit {
				continue
			}

			start, end := s.Start, s.End
			// Widen JSON escapes if applicable
			start, end = widenJSONEscapes(text, start, end)

			// Clip span end at inspectLimit if it crosses
			if end > inspectLimit {
				end = inspectLimit
			}

			if start < lastEnd {
				start = lastEnd
			}
			if start >= end {
				continue
			}

			pl := s.Label.Placeholder()
			if pl == "" {
				pl = types.RedactedUninspected
			}

			processed = append(processed, validSpan{
				start:       start,
				end:         end,
				placeholder: pl,
			})
			lastEnd = end
		}
	}

	if len(processed) == 0 && !isOversize {
		return text, nil
	}

	// Compute required output capacity
	estimatedCap := textLen
	for _, p := range processed {
		estimatedCap += len(p.placeholder) - (p.end - p.start)
	}
	if isOversize && mode == types.ErrorModeDrop {
		estimatedCap += len(types.RedactedUninspected) - (textLen - inspectLimit)
	}
	if estimatedCap < 0 {
		estimatedCap = 0
	}

	var buf []byte
	if scratch != nil && cap(scratch) >= estimatedCap {
		buf = scratch[:0]
	} else {
		buf = make([]byte, 0, estimatedCap)
	}

	lastIdx := 0
	for _, p := range processed {
		buf = append(buf, text[lastIdx:p.start]...)
		buf = append(buf, p.placeholder...)
		lastIdx = p.end
	}

	if isOversize {
		buf = append(buf, text[lastIdx:inspectLimit]...)
		if mode == types.ErrorModeDrop {
			buf = append(buf, types.RedactedUninspected...)
		} else {
			buf = append(buf, text[inspectLimit:]...)
		}
	} else {
		buf = append(buf, text[lastIdx:]...)
	}

	return string(buf), nil
}

func isRuneBoundary(s string, idx int) bool {
	if idx <= 0 || idx >= len(s) {
		return true
	}
	// A continuation byte in UTF-8 has the top bits 10xxxxxx (0x80..0xBF).
	return (s[idx] & 0xC0) != 0x80
}

func isHexDigit(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F')
}

func isValidUnicodeEscape(text string, idx int) bool {
	if idx < 0 || idx+6 > len(text) || text[idx] != '\\' || text[idx+1] != 'u' {
		return false
	}
	// Ensure the backslash at idx is not itself escaped by a preceding backslash
	bCount := 0
	for i := idx; i >= 0 && text[i] == '\\'; i-- {
		bCount++
	}
	if bCount%2 == 0 {
		return false
	}
	return isHexDigit(text[idx+2]) && isHexDigit(text[idx+3]) && isHexDigit(text[idx+4]) && isHexDigit(text[idx+5])
}

func widenJSONEscapes(text string, start, end int) (int, int) {
	// 1. Check if start cuts into a unicode escape \uXXXX
	for back := 1; back <= 5 && start-back >= 0; back++ {
		idx := start - back
		if isValidUnicodeEscape(text, idx) {
			if start < idx+6 && idx+6 <= len(text) {
				start = idx
				if end < idx+6 {
					end = idx + 6
				}
			}
			break
		}
	}

	// 2. Check if start landed on an escaped quote (e.g. \")
	if start > 0 && start < len(text) && text[start] == '"' {
		bCount := 0
		for i := start - 1; i >= 0 && text[i] == '\\'; i-- {
			bCount++
		}
		if bCount%2 == 1 {
			start--
		}
	}

	// 3. Check if end cuts into \uXXXX
	for back := 1; back <= 5 && end-back >= 0; back++ {
		idx := end - back
		if isValidUnicodeEscape(text, idx) {
			if end < idx+6 && idx+6 <= len(text) {
				end = idx + 6
			}
			break
		}
	}

	// 4. Check if end landed on a backslash escaping a quote (e.g. \" where end is at \")
	if end > 0 && end < len(text) && text[end] == '"' && text[end-1] == '\\' {
		bCount := 0
		for i := end - 1; i >= 0 && text[i] == '\\'; i-- {
			bCount++
		}
		if bCount%2 == 1 {
			end++
		}
	}

	return start, end
}
