package mask

import (
	"testing"
	"unicode/utf8"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func FuzzMask(f *testing.F) {
	seeds := []string{
		"Hello world",
		"/home/alice/.ssh/id_rsa",
		`{"key": "value \"inside\" \u0041"}`,
		"日本語とEnglish mixed text",
		"Clean text",
	}

	for _, s := range seeds {
		f.Add(s, 0, len(s), uint8(types.LabelPerson), 100, uint8(0))
	}

	f.Fuzz(func(t *testing.T, text string, start, end int, labelVal uint8, maxBytes int, modeVal uint8) {
		if !utf8.ValidString(text) {
			return
		}
		if start < 0 || end < 0 || start > len(text) || end > len(text) || start > end {
			return
		}

		lbl := types.Label(labelVal % 10)
		mode := types.ErrorModeDrop
		if modeVal%2 == 1 {
			mode = types.ErrorModeIgnore
		}

		var spans []types.EntitySpan
		if start < end && lbl != types.LabelNone {
			spans = []types.EntitySpan{
				{
					Start:      start,
					End:        end,
					Label:      lbl,
					Confidence: 0.9,
				},
			}
		}

		out, err := Mask(text, spans, maxBytes, mode, nil)
		if err == nil {
			if !utf8.ValidString(out) {
				t.Fatalf("Mask output is not valid UTF-8: %q", out)
			}
		}
	})
}
