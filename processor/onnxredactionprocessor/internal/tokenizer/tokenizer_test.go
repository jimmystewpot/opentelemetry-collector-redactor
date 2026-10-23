package tokenizer

import (
	"path/filepath"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

const (
	validTokenizerPath = "../../testdata/tokenizer.json"
	validSHA256        = "094033eb54d75f1d08c3726a0f0a0ab42c92ccb0f22bd27f4e9ed2d1caccd2fa"
)

func TestNewWordPieceTokenizer_ChecksumValidation(t *testing.T) {
	// 1. Valid checksum
	tok, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	require.NotNil(t, tok)
	defer func() { _ = tok.Close() }()

	// 2. Empty checksum (should bypass checksum check)
	tok2, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), "")
	require.NoError(t, err)
	defer func() { _ = tok2.Close() }()

	// 3. Mismatched checksum
	_, err = NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
	require.Error(t, err, "must reject invalid SHA-256")

	// 4. Non-existent file
	_, err = NewWordPieceTokenizer("non_existent_file.json", "")
	require.Error(t, err)
}

func TestWordPieceTokenizer_EncodeOffsetsAndRuneBoundaries(t *testing.T) {
	tok, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tok.Close() }()

	cases := []struct {
		name string
		text string
	}{
		{"ASCII sentence", "the user alice requested access to api v1"},
		{"Accented Latin", "Marie Müller was in Zürich"},
		{"CJK characters", "東京の山田太郎"},
		{"Emoji and symbols", "key 🔑 path /home/alice/secret"},
		{"File path", "/Users/marie.curie/.ssh/id_rsa"},
		{"Windows JSON-escaped", `C:\\Users\\Administrator\\secret.log`},
		{"Empty string", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var ids []int64
			var offsets []types.Offset

			ids, offsets, err := tok.Encode(tc.text, ids, offsets)
			require.NoError(t, err)

			// First token must be [CLS] with offset (0, 0)
			require.NotEmpty(t, ids)
			assert.Equal(t, int64(101), ids[0], "[CLS] token ID")
			assert.Equal(t, types.Offset{Start: 0, End: 0}, offsets[0])

			// Last token must be [SEP] with offset (len(text), len(text))
			lastIdx := len(ids) - 1
			assert.Equal(t, int64(102), ids[lastIdx], "[SEP] token ID")
			assert.Equal(t, types.Offset{Start: len(tc.text), End: len(tc.text)}, offsets[lastIdx])

			// Every offset must be within [0, len(text)] and on valid UTF-8 rune boundaries
			for i, off := range offsets {
				assert.GreaterOrEqual(t, off.Start, 0, "offset %d start >= 0", i)
				assert.LessOrEqual(t, off.End, len(tc.text), "offset %d end <= len", i)
				assert.LessOrEqual(t, off.Start, off.End, "offset %d start <= end", i)

				if off.Start < len(tc.text) {
					assert.True(t, utf8.RuneStart(tc.text[off.Start]), "off.Start must be a rune boundary")
				}
				if off.End < len(tc.text) {
					assert.True(t, utf8.RuneStart(tc.text[off.End]), "off.End must be a rune boundary")
				}
			}
		})
	}
}

func TestWordPieceTokenizer_TokenIDsMatching(t *testing.T) {
	tok, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tok.Close() }()

	// Text contains known tokens: "the user alice"
	// "the" -> 300, "user" -> 313, "alice" -> 314
	ids, offsets, err := tok.Encode("the user alice", nil, nil)
	require.NoError(t, err)

	// [CLS], "the", "user", "alice", [SEP]
	require.Len(t, ids, 5)
	assert.Equal(t, int64(101), ids[0])
	assert.Equal(t, int64(300), ids[1])
	assert.Equal(t, int64(313), ids[2])
	assert.Equal(t, int64(314), ids[3])
	assert.Equal(t, int64(102), ids[4])

	assert.Equal(t, types.Offset{Start: 0, End: 3}, offsets[1])  // "the"
	assert.Equal(t, types.Offset{Start: 4, End: 8}, offsets[2])  // "user"
	assert.Equal(t, types.Offset{Start: 9, End: 14}, offsets[3]) // "alice"
}
