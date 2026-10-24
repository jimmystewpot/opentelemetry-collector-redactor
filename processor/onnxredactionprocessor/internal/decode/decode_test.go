package decode

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

var defaultBIO = []string{
	"O",
	"B-PERSON", "B-EMAIL", "B-PHONE", "B-LOCATION", "B-CREDIT_CARD", "B-SSN", "B-IBAN", "B-MEDICAL_ID", "B-PATH_USER",
	"I-PERSON", "I-EMAIL", "I-PHONE", "I-LOCATION", "I-CREDIT_CARD", "I-SSN", "I-IBAN", "I-MEDICAL_ID", "I-PATH_USER",
}

func TestDecode_DecodeWindow(t *testing.T) {
	d := NewDecoder(defaultBIO, nil, 0.85)

	numClasses := 19
	// 4 tokens:
	// Token 0: [CLS], offset (0, 0) -> special token, should be ignored even if high logit
	// Token 1: "John", offset (0, 4) -> B-PERSON (class 1)
	// Token 2: "Doe", offset (5, 8) -> I-PERSON (class 10)
	// Token 3: [SEP], offset (8, 8) -> special token
	offsets := []types.Offset{
		{Start: 0, End: 0},
		{Start: 0, End: 4},
		{Start: 5, End: 8},
		{Start: 8, End: 8},
	}

	logits := make([]float32, 4*numClasses)
	// Token 0: class 1 has 10.0
	logits[0*numClasses+1] = 10.0
	// Token 1: class 1 (B-PERSON) has 10.0, others 0.0 -> prob ~0.999
	logits[1*numClasses+1] = 10.0
	// Token 2: class 10 (I-PERSON) has 10.0, others 0.0 -> prob ~0.999
	logits[2*numClasses+10] = 10.0
	// Token 3: class 10 has 10.0
	logits[3*numClasses+10] = 10.0

	var dst []types.EntitySpan
	spans := d.DecodeWindow(logits, numClasses, offsets, dst)
	require.Len(t, spans, 1)

	// John + Doe merged into single [0, 8) PERSON span
	assert.Equal(t, 0, spans[0].Start)
	assert.Equal(t, 8, spans[0].End)
	assert.Equal(t, types.LabelPerson, spans[0].Label)
	assert.Greater(t, spans[0].Confidence, float32(0.85))
}

func TestDecode_ThresholdFiltering(t *testing.T) {
	// Custom threshold: PERSON requires 0.99
	thresholds := map[types.Label]float32{
		types.LabelPerson: 0.99,
	}
	d := NewDecoder(defaultBIO, thresholds, 0.85)

	numClasses := 19
	offsets := []types.Offset{{Start: 0, End: 4}}
	logits := make([]float32, numClasses)
	// Set class 1 (B-PERSON) to 2.0, class 0 to 1.0 -> prob is ~0.73, below 0.99
	logits[1] = 2.0
	logits[0] = 1.0

	spans := d.DecodeWindow(logits, numClasses, offsets, nil)
	assert.Empty(t, spans, "should be filtered out by confidence threshold")
}

func TestMergeSpans_SameLabelUnion(t *testing.T) {
	// Overlapping spans of same label: [10, 25) and [20, 35) -> [10, 35)
	spans := []types.EntitySpan{
		{Start: 10, End: 25, Label: types.LabelPerson, Confidence: 0.90},
		{Start: 20, End: 35, Label: types.LabelPerson, Confidence: 0.95},
	}

	merged := MergeSpans(spans)
	require.Len(t, merged, 1)
	assert.Equal(t, 10, merged[0].Start)
	assert.Equal(t, 35, merged[0].End)
	assert.Equal(t, types.LabelPerson, merged[0].Label)
	assert.Equal(t, float32(0.95), merged[0].Confidence)
}

func TestMergeSpans_AdjacentSameLabelUnion(t *testing.T) {
	// Adjacent spans of same label: [0, 5) and [5, 10) -> [0, 10)
	spans := []types.EntitySpan{
		{Start: 0, End: 5, Label: types.LabelEmail, Confidence: 0.92},
		{Start: 5, End: 10, Label: types.LabelEmail, Confidence: 0.94},
	}

	merged := MergeSpans(spans)
	require.Len(t, merged, 1)
	assert.Equal(t, 0, merged[0].Start)
	assert.Equal(t, 10, merged[0].End)
}

func TestMergeSpans_ConflictingLabels(t *testing.T) {
	// Overlapping spans of different labels:
	// Span 1: [10, 30) LabelPerson, Conf 0.88
	// Span 2: [20, 40) LabelEmail, Conf 0.96 (higher confidence wins)
	spans := []types.EntitySpan{
		{Start: 10, End: 30, Label: types.LabelPerson, Confidence: 0.88},
		{Start: 20, End: 40, Label: types.LabelEmail, Confidence: 0.96},
	}

	merged := MergeSpans(spans)
	require.Len(t, merged, 1)
	assert.Equal(t, types.LabelEmail, merged[0].Label)
	assert.Equal(t, float32(0.96), merged[0].Confidence)
}

func TestMergeSpans_TieBreakByLabelOrder(t *testing.T) {
	// Equal confidence, different labels:
	// LabelPerson (enum 1) vs LabelLocation (enum 4): LabelPerson should win
	spans := []types.EntitySpan{
		{Start: 0, End: 20, Label: types.LabelLocation, Confidence: 0.90},
		{Start: 10, End: 30, Label: types.LabelPerson, Confidence: 0.90},
	}

	merged := MergeSpans(spans)
	require.Len(t, merged, 1)
	assert.Equal(t, types.LabelPerson, merged[0].Label)
}

func TestDecode_EdgeCasesAndCoverage(t *testing.T) {
	d := NewDecoder([]string{"O", "B-PERSON", "I-PERSON", "B-EMAIL", "UNKNOWN"}, nil, 0)
	assert.Equal(t, float32(0.85), d.defaultThreshold)

	// Empty / invalid arguments
	assert.Nil(t, d.DecodeWindow(nil, 5, nil, nil))
	assert.Nil(t, d.DecodeWindow([]float32{1.0}, 0, []types.Offset{{Start: 0, End: 1}}, nil))
	assert.Nil(t, d.DecodeWindow([]float32{1.0}, 5, []types.Offset{{Start: 0, End: 1}}, nil))

	// Standalone I- tag without preceding B-
	offsets := []types.Offset{{Start: 0, End: 4}}
	logits := []float32{0, 0, 10.0, 0, 0} // index 2 is I-PERSON
	spans := d.DecodeWindow(logits, 5, offsets, nil)
	require.Len(t, spans, 1)
	assert.Equal(t, types.LabelPerson, spans[0].Label)

	// B- tag immediately following another B- tag
	offsets2 := []types.Offset{{Start: 0, End: 4}, {Start: 5, End: 15}}
	logits2 := []float32{
		0, 10.0, 0, 0, 0, // B-PERSON
		0, 0, 0, 10.0, 0, // B-EMAIL
	}
	spans2 := d.DecodeWindow(logits2, 5, offsets2, nil)
	require.Len(t, spans2, 2)
	assert.Equal(t, types.LabelPerson, spans2[0].Label)
	assert.Equal(t, types.LabelEmail, spans2[1].Label)

	// MergeSpans edge cases
	assert.Empty(t, MergeSpans(nil))
	assert.Len(t, MergeSpans([]types.EntitySpan{{Start: 0, End: 5, Label: types.LabelPerson}}), 1)
	assert.Empty(t, MergeSpans([]types.EntitySpan{{Start: 5, End: 5, Label: types.LabelPerson}})) // 0-length

	// Encompassing span with higher confidence
	encompassing := []types.EntitySpan{
		{Start: 0, End: 20, Label: types.LabelPerson, Confidence: 0.95},
		{Start: 5, End: 15, Label: types.LabelLocation, Confidence: 0.80},
	}
	mEncomp := MergeSpans(encompassing)
	require.Len(t, mEncomp, 1)
	assert.Equal(t, types.LabelPerson, mEncomp[0].Label)
	assert.Equal(t, 0, mEncomp[0].Start)
	assert.Equal(t, 20, mEncomp[0].End)
}

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
