package mask

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func spanOf(haystack, needle string, label types.Label) types.EntitySpan {
	idx := strings.Index(haystack, needle)
	if idx == -1 {
		panic("needle not found in haystack: " + needle)
	}
	return types.EntitySpan{
		Start:      idx,
		End:        idx + len(needle),
		Label:      label,
		Confidence: 0.99,
	}
}

func TestMask_CleanString(t *testing.T) {
	text := "Clean log message with no entities."
	var out string
	var err error
	allocs := testing.AllocsPerRun(10, func() {
		out, err = Mask(text, nil, 16384, types.ErrorModeDrop, nil)
	})
	require.NoError(t, err)
	assert.Equal(t, text, out)
	assert.Equal(t, float64(0), allocs, "clean string should have 0 allocs/op")
}

func TestMask_PathUser(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		user     string
		expected string
	}{
		{
			name:     "linux home",
			input:    "/home/jdoe/.ssh/id_rsa",
			user:     "jdoe",
			expected: "/home/[REDACTED_PATH_USER]/.ssh/id_rsa",
		},
		{
			name:     "macos users",
			input:    "/Users/marie.curie/Documents/db.sqlite",
			user:     "marie.curie",
			expected: "/Users/[REDACTED_PATH_USER]/Documents/db.sqlite",
		},
		{
			name:     "windows users",
			input:    `C:\Users\Administrator\secret.log`,
			user:     "Administrator",
			expected: `C:\Users\[REDACTED_PATH_USER]\secret.log`,
		},
		{
			name:     "windows json-escaped",
			input:    `C:\\Users\\Administrator\\secret.log`,
			user:     "Administrator",
			expected: `C:\\Users\\[REDACTED_PATH_USER]\\secret.log`,
		},
		{
			name:     "unc path",
			input:    `\\filesrv\homes\jdoe\payroll.xlsx`,
			user:     "jdoe",
			expected: `\\filesrv\homes\[REDACTED_PATH_USER]\payroll.xlsx`,
		},
		{
			name:     "wsl path",
			input:    "/mnt/c/Users/jdoe/keys",
			user:     "jdoe",
			expected: "/mnt/c/Users/[REDACTED_PATH_USER]/keys",
		},
		{
			name:     "file uri",
			input:    "file:///home/service_user/credentials.env",
			user:     "service_user",
			expected: "file:///home/[REDACTED_PATH_USER]/credentials.env",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			span := spanOf(tt.input, tt.user, types.LabelPathUser)
			out, err := Mask(tt.input, []types.EntitySpan{span}, 16384, types.ErrorModeDrop, nil)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, out)
		})
	}
}

func TestMask_MultiByteAndRuneBoundaries(t *testing.T) {
	input := "/Users/müller/日本語/🔑.pem"
	span := spanOf(input, "müller", types.LabelPathUser)
	out, err := Mask(input, []types.EntitySpan{span}, 16384, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.Equal(t, "/Users/[REDACTED_PATH_USER]/日本語/🔑.pem", out)

	// Mid-rune invalid span: 'ü' is 2 bytes (C3 BC). Start in middle of 'ü'
	uIdx := strings.Index(input, "ü")
	badSpan := types.EntitySpan{
		Start:      uIdx + 1, // mid-rune!
		End:        uIdx + 2,
		Label:      types.LabelPerson,
		Confidence: 0.9,
	}
	_, err = Mask(input, []types.EntitySpan{badSpan}, 16384, types.ErrorModeDrop, nil)
	require.Error(t, err, "should reject mid-rune span boundary")
}

func TestMask_JSONEscapeBoundaryWidening(t *testing.T) {
	// JSON document where a string contains an escaped quote and unicode escape
	jsonStr := `{"msg": "user \"alice\" logged in from \u0041"}`
	require.True(t, json.Valid([]byte(jsonStr)))

	// Suppose detector identifies "alice"
	span1 := spanOf(jsonStr, "alice", types.LabelPerson)
	out1, err := Mask(jsonStr, []types.EntitySpan{span1}, 16384, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.True(t, json.Valid([]byte(out1)), "output must remain valid JSON: %s", out1)

	// Suppose detector flagged the unicode escape \u0041 starting inside the escape
	uIdx := strings.Index(jsonStr, `\u0041`)
	span2 := types.EntitySpan{
		Start:      uIdx + 1, // points to 'u' instead of '\'
		End:        uIdx + 6,
		Label:      types.LabelPerson,
		Confidence: 0.9,
	}
	out2, err := Mask(jsonStr, []types.EntitySpan{span2}, 16384, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.True(t, json.Valid([]byte(out2)), "widened escape must preserve valid JSON: %s", out2)
}

func TestMask_OversizeTail(t *testing.T) {
	longPrefix := strings.Repeat("A", 100)
	secret := "John Doe"
	longSuffix := strings.Repeat("Z", 100)
	input := longPrefix + secret + longSuffix

	span := spanOf(input, secret, types.LabelPerson)

	// Cap at 150 bytes (secret starts at 100 and ends at 108, so secret is inside inspected prefix)
	maxBytes := 150

	// Under ErrorModeDrop: tail after 150 bytes is replaced by [REDACTED_UNINSPECTED]
	outDrop, err := Mask(input, []types.EntitySpan{span}, maxBytes, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.Contains(t, outDrop, "[REDACTED_PERSON]")
	assert.True(t, strings.HasSuffix(outDrop, types.RedactedUninspected))

	// Under ErrorModeIgnore: tail after 150 bytes is preserved unchanged
	outIgnore, err := Mask(input, []types.EntitySpan{span}, maxBytes, types.ErrorModeIgnore, nil)
	require.NoError(t, err)
	assert.Contains(t, outIgnore, "[REDACTED_PERSON]")
	assert.False(t, strings.Contains(outIgnore, types.RedactedUninspected))
	assert.True(t, strings.HasSuffix(outIgnore, longSuffix))
}

func TestMask_InvalidUTF8(t *testing.T) {
	invalidUTF8 := string([]byte{0xff, 0xfe, 0xfd})
	_, err := Mask(invalidUTF8, nil, 100, types.ErrorModeDrop, nil)
	require.Error(t, err)
}

func TestMask_BoundsAndOverlapErrors(t *testing.T) {
	text := "Hello John and Jane"

	// Out of bounds
	_, err := Mask(text, []types.EntitySpan{{Start: -1, End: 5}}, 100, types.ErrorModeDrop, nil)
	require.Error(t, err)

	_, err = Mask(text, []types.EntitySpan{{Start: 0, End: 100}}, 100, types.ErrorModeDrop, nil)
	require.Error(t, err)

	_, err = Mask(text, []types.EntitySpan{{Start: 5, End: 2}}, 100, types.ErrorModeDrop, nil)
	require.Error(t, err)

	// Overlapping spans
	s1 := spanOf(text, "John", types.LabelPerson)
	sOverlap := types.EntitySpan{Start: s1.Start + 1, End: s1.End + 5, Label: types.LabelPerson}
	_, err = Mask(text, []types.EntitySpan{s1, sOverlap}, 100, types.ErrorModeDrop, nil)
	require.Error(t, err)
}

func TestMask_ScratchAndCrossLimit(t *testing.T) {
	text := "User alice is in admin role"
	s := spanOf(text, "alice", types.LabelPerson)

	// Use scratch buffer
	scratch := make([]byte, 256)
	out, err := Mask(text, []types.EntitySpan{s}, 100, types.ErrorModeDrop, scratch)
	require.NoError(t, err)
	assert.Equal(t, "User [REDACTED_PERSON] is in admin role", out)

	// Span entirely beyond maxPayloadBytes
	longText := strings.Repeat("A", 100) + "Bob"
	sBeyond := spanOf(longText, "Bob", types.LabelPerson)
	outBeyond, err := Mask(longText, []types.EntitySpan{sBeyond}, 50, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.False(t, strings.Contains(outBeyond, "[REDACTED_PERSON]"))
	assert.True(t, strings.HasSuffix(outBeyond, types.RedactedUninspected))

	// Span crossing inspectLimit
	crossingText := "Prefix 1234567890 JohnDoe 123"
	// limit right in the middle of JohnDoe
	limit := strings.Index(crossingText, "JohnDoe") + 4
	sCross := spanOf(crossingText, "JohnDoe", types.LabelPerson)
	outCross, err := Mask(crossingText, []types.EntitySpan{sCross}, limit, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.Contains(t, outCross, "[REDACTED_PERSON]")
	assert.True(t, strings.HasSuffix(outCross, types.RedactedUninspected))

	// Empty placeholder fallback
	sNone := spanOf(text, "alice", types.LabelNone)
	outNone, err := Mask(text, []types.EntitySpan{sNone}, 100, types.ErrorModeDrop, nil)
	require.NoError(t, err)
	assert.Contains(t, outNone, types.RedactedUninspected)
}

func TestWidenJSONEscapes_Direct(t *testing.T) {
	// Escaped quote: \"text\"
	text := `abc\"def\"ghi`
	// start on quote: index 4 is '"', index 3 is '\'
	s, e := widenJSONEscapes(text, 4, 7)
	assert.Equal(t, 3, s)
	assert.Equal(t, 7, e)

	// end on quote: index 9 is '"', index 8 is '\'
	s2, e2 := widenJSONEscapes(text, 5, 9)
	assert.Equal(t, 5, s2)
	assert.Equal(t, 10, e2)

	// Unicode escape: \u0041
	uText := `prefix \u0041 suffix`
	// start cuts into unicode escape
	s3, e3 := widenJSONEscapes(uText, 9, 12)
	assert.Equal(t, 7, s3)
	assert.Equal(t, 13, e3)

	// end cuts into unicode escape
	s4, e4 := widenJSONEscapes(uText, 2, 10)
	assert.Equal(t, 2, s4)
	assert.Equal(t, 13, e4)
}
