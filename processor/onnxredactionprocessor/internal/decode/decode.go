package decode

import (
	"math"
	"sort"
	"strings"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// Decoder converts per-token class logits into EntitySpans based on BIO tagging.
type Decoder struct {
	classToLabel     []types.Label
	isB              []bool
	isI              []bool
	thresholds       map[types.Label]float32
	defaultThreshold float32
}

// NewDecoder initializes a Decoder with pre-parsed BIO class tables and threshold mappings.
func NewDecoder(idToBIO []string, thresholds map[types.Label]float32, defaultThreshold float32) *Decoder {
	if defaultThreshold <= 0 {
		defaultThreshold = 0.85
	}

	n := len(idToBIO)
	classToLabel := make([]types.Label, n)
	isB := make([]bool, n)
	isI := make([]bool, n)

	for i, tag := range idToBIO {
		tag = strings.TrimSpace(tag)
		switch {
		case strings.HasPrefix(tag, "B-"):
			lbl, _ := types.ParseLabel(strings.TrimPrefix(tag, "B-"))
			classToLabel[i] = lbl
			isB[i] = true
		case strings.HasPrefix(tag, "I-"):
			lbl, _ := types.ParseLabel(strings.TrimPrefix(tag, "I-"))
			classToLabel[i] = lbl
			isI[i] = true
		default:
			classToLabel[i] = types.LabelNone
		}
	}

	return &Decoder{
		classToLabel:     classToLabel,
		isB:              isB,
		isI:              isI,
		thresholds:       thresholds,
		defaultThreshold: defaultThreshold,
	}
}

func (d *Decoder) thresholdFor(lbl types.Label) float32 {
	if d.thresholds != nil {
		if th, ok := d.thresholds[lbl]; ok {
			return th
		}
	}
	return d.defaultThreshold
}

// DecodeWindow converts per-token logits and token byte offsets into parent-string EntitySpans.
func (d *Decoder) DecodeWindow(logits []float32, numClasses int, offsets []types.Offset, dst []types.EntitySpan) []types.EntitySpan {
	seqLen := len(offsets)
	if seqLen == 0 || numClasses <= 0 || len(logits) < seqLen*numClasses {
		return dst
	}

	var current types.EntitySpan
	hasCurrent := false

	for t := 0; t < seqLen; t++ {
		off := offsets[t]
		// Skip special / padding tokens with (0,0) offset
		if off.Start == off.End {
			continue
		}

		offsetIdx := t * numClasses
		row := logits[offsetIdx : offsetIdx+numClasses]

		// Find max logit and argmax
		maxLogit := float32(-math.MaxFloat32)
		bestClass := 0
		for c, val := range row {
			if val > maxLogit {
				maxLogit = val
				bestClass = c
			}
		}

		// Compute softmax denominator
		sumExp := float32(0.0)
		for _, val := range row {
			sumExp += float32(math.Exp(float64(val - maxLogit)))
		}

		prob := float32(1.0)
		if sumExp > 0 {
			prob = float32(math.Exp(float64(row[bestClass]-maxLogit))) / sumExp
		}

		lbl := types.LabelNone
		if bestClass >= 0 && bestClass < len(d.classToLabel) {
			lbl = d.classToLabel[bestClass]
		}
		th := d.thresholdFor(lbl)

		if lbl == types.LabelNone || prob < th {
			// Close current entity if any
			if hasCurrent {
				dst = append(dst, current)
				hasCurrent = false
			}
			continue
		}

		if d.isB[bestClass] {
			if hasCurrent {
				dst = append(dst, current)
			}
			current = types.EntitySpan{
				Start:      off.Start,
				End:        off.End,
				Label:      lbl,
				Confidence: prob,
			}
			hasCurrent = true
		} else if d.isI[bestClass] {
			if hasCurrent && current.Label == lbl {
				current.End = off.End
				if prob > current.Confidence {
					current.Confidence = prob
				}
			} else {
				if hasCurrent {
					dst = append(dst, current)
				}
				current = types.EntitySpan{
					Start:      off.Start,
					End:        off.End,
					Label:      lbl,
					Confidence: prob,
				}
				hasCurrent = true
			}
		}
	}

	if hasCurrent {
		dst = append(dst, current)
	}

	return dst
}

// MergeSpans merges overlapping or adjacent spans of the same Label and resolves
// conflicting labels by higher Confidence (tie-breaking by lower Label enum value).
func MergeSpans(spans []types.EntitySpan) []types.EntitySpan {
	if len(spans) == 0 {
		return nil
	}
	// Filter out zero-length spans first
	var valid []types.EntitySpan
	var stackValid [16]types.EntitySpan
	if len(spans) <= 16 {
		valid = stackValid[:0]
	} else {
		valid = make([]types.EntitySpan, 0, len(spans))
	}
	for _, s := range spans {
		if s.Start < s.End {
			valid = append(valid, s)
		}
	}
	if len(valid) <= 1 {
		if len(valid) == 0 {
			return nil
		}
		res := make([]types.EntitySpan, len(valid))
		copy(res, valid)
		return res
	}

	// Sort spans by Start asc, then End desc
	sort.Slice(valid, func(i, j int) bool {
		if valid[i].Start != valid[j].Start {
			return valid[i].Start < valid[j].Start
		}
		return valid[i].End > valid[j].End
	})

	result := make([]types.EntitySpan, 0, len(valid))
	current := valid[0]

	for i := 1; i < len(valid); i++ {
		next := valid[i]

		// Overlap or adjacency check
		if next.Start <= current.End {
			if next.Label == current.Label {
				// Same label union
				if next.End > current.End {
					current.End = next.End
				}
				if next.Confidence > current.Confidence {
					current.Confidence = next.Confidence
				}
			} else {
				// Different label conflict resolution: higher-confidence label wins
				currentWins := current.Confidence > next.Confidence ||
					(current.Confidence == next.Confidence && current.Label <= next.Label)

				if currentWins {
					if next.End > current.End {
						current.End = next.End
					}
				} else {
					// next wins
					if current.Start < next.Start {
						next.Start = current.Start
					}
					if current.End > next.End {
						next.End = current.End
					}
					current = next
				}
			}
		} else {
			result = append(result, current)
			current = next
		}
	}

	result = append(result, current)
	return result
}
