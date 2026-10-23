package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLabelStringAndPlaceholder(t *testing.T) {
	tests := []struct {
		label       Label
		expectedStr string
		expectedPl  string
	}{
		{LabelNone, "NONE", ""},
		{LabelPerson, "PERSON", "[REDACTED_PERSON]"},
		{LabelEmail, "EMAIL", "[REDACTED_EMAIL]"},
		{LabelPhone, "PHONE", "[REDACTED_PHONE]"},
		{LabelLocation, "LOCATION", "[REDACTED_LOCATION]"},
		{LabelCreditCard, "CREDIT_CARD", "[REDACTED_CREDIT_CARD]"},
		{LabelSSN, "SSN", "[REDACTED_SSN]"},
		{LabelIBAN, "IBAN", "[REDACTED_IBAN]"},
		{LabelMedicalID, "MEDICAL_ID", "[REDACTED_MEDICAL_ID]"},
		{LabelPathUser, "PATH_USER", "[REDACTED_PATH_USER]"},
	}

	for _, tt := range tests {
		t.Run(tt.expectedStr, func(t *testing.T) {
			assert.Equal(t, tt.expectedStr, tt.label.String())
			assert.Equal(t, tt.expectedPl, tt.label.Placeholder())
		})
	}
}

func TestParseLabel(t *testing.T) {
	tests := []struct {
		input    string
		expected Label
		ok       bool
	}{
		{"PERSON", LabelPerson, true},
		{"person", LabelPerson, true},
		{"EMAIL", LabelEmail, true},
		{"PHONE", LabelPhone, true},
		{"LOCATION", LabelLocation, true},
		{"CREDIT_CARD", LabelCreditCard, true},
		{"SSN", LabelSSN, true},
		{"IBAN", LabelIBAN, true},
		{"MEDICAL_ID", LabelMedicalID, true},
		{"PATH_USER", LabelPathUser, true},
		{"UNKNOWN", LabelNone, false},
		{"", LabelNone, false},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			lbl, ok := ParseLabel(tt.input)
			assert.Equal(t, tt.ok, ok)
			assert.Equal(t, tt.expected, lbl)
		})
	}
}

func TestConstantsAndStructs(t *testing.T) {
	require.Equal(t, "[REDACTED_UNINSPECTED]", RedactedUninspected)

	span := EntitySpan{
		Start:      0,
		End:        5,
		Label:      LabelPerson,
		Confidence: 0.95,
	}
	assert.Equal(t, 0, span.Start)
	assert.Equal(t, 5, span.End)
	assert.Equal(t, LabelPerson, span.Label)
	assert.InDelta(t, 0.95, span.Confidence, 0.001)

	offset := Offset{Start: 0, End: 4}
	assert.Equal(t, 0, offset.Start)
	assert.Equal(t, 4, offset.End)

	tw := TokenWindow{
		ParentUniqueIdx: 1,
		TokenIDs:        []int64{101, 200, 102},
		Offsets:         []Offset{{Start: 0, End: 0}, {Start: 0, End: 4}, {Start: 4, End: 4}},
	}
	assert.Equal(t, 1, tw.ParentUniqueIdx)
	assert.Len(t, tw.TokenIDs, 3)
	assert.Len(t, tw.Offsets, 3)

	bb := BucketedBatch{
		MaxSeqLen: 32,
		Windows:   []TokenWindow{tw},
	}
	assert.Equal(t, 32, bb.MaxSeqLen)
	assert.Len(t, bb.Windows, 1)

	assert.Equal(t, SignalKind(0), SignalLogs)
	assert.Equal(t, SignalKind(1), SignalTraces)
	assert.Equal(t, SignalKind(2), SignalMetrics)

	assert.Equal(t, ErrorMode("drop"), ErrorModeDrop)
	assert.Equal(t, ErrorMode("ignore"), ErrorModeIgnore)
}
