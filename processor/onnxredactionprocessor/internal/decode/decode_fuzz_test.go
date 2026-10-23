package decode

import (
	"testing"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func FuzzMergeSpans(f *testing.F) {
	f.Add(0, 10, uint8(1), float32(0.9), 5, 15, uint8(1), float32(0.8))
	f.Add(10, 20, uint8(1), float32(0.8), 15, 25, uint8(2), float32(0.95))
	f.Add(0, 5, uint8(2), float32(0.9), 5, 10, uint8(2), float32(0.9))

	f.Fuzz(func(t *testing.T, s1, e1 int, l1 uint8, c1 float32, s2, e2 int, l2 uint8, c2 float32) {
		if s1 < 0 || e1 < 0 || s2 < 0 || e2 < 0 || s1 > e1 || s2 > e2 {
			return
		}
		if s1 > 10000 || e1 > 10000 || s2 > 10000 || e2 > 10000 {
			return
		}

		lbl1 := types.Label(l1 % 10)
		lbl2 := types.Label(l2 % 10)
		if lbl1 == types.LabelNone || lbl2 == types.LabelNone {
			return
		}

		spans := []types.EntitySpan{
			{Start: s1, End: e1, Label: lbl1, Confidence: c1},
			{Start: s2, End: e2, Label: lbl2, Confidence: c2},
		}

		merged := MergeSpans(spans)

		// Invariant: merged spans must be strictly non-overlapping and sorted
		for i := 0; i < len(merged); i++ {
			if merged[i].Start > merged[i].End {
				t.Fatalf("invalid inverted span [%d, %d]", merged[i].Start, merged[i].End)
			}
			if i > 0 && merged[i-1].End > merged[i].Start {
				t.Fatalf("overlapping spans in merged output: [%d, %d] and [%d, %d]",
					merged[i-1].Start, merged[i-1].End, merged[i].Start, merged[i].End)
			}
		}
	})
}
