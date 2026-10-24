package tokenizer

import (
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

const (
	validTokenizerPath = "../../testdata/tokenizer.json"
	validSHA256        = "2afe60ae310a17a1807c43abe1b6867b8b2985ca44bced09a11da3c90d227884"
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

func TestWordPieceTokenizer_HyphenAndUnderscoreSplitting(t *testing.T) {
	tok, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tok.Close() }()

	// "alice-smith" -> [CLS](101), "alice"(314), "-"(367), "smith"(318), [SEP](102)
	ids, offsets, err := tok.Encode("alice-smith", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []int64{101, 314, 367, 318, 102}, ids)
	assert.Equal(t, []types.Offset{
		{Start: 0, End: 0},
		{Start: 0, End: 5},
		{Start: 5, End: 6},
		{Start: 6, End: 11},
		{Start: 11, End: 11},
	}, offsets)

	// "alice_smith" -> [CLS](101), "alice"(314), "_"(368), "smith"(318), [SEP](102)
	idsUnder, offsetsUnder, err := tok.Encode("alice_smith", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []int64{101, 314, 368, 318, 102}, idsUnder)
	assert.Equal(t, []types.Offset{
		{Start: 0, End: 0},
		{Start: 0, End: 5},
		{Start: 5, End: 6},
		{Start: 6, End: 11},
		{Start: 11, End: 11},
	}, offsetsUnder)
}

func TestWordPieceTokenizer_ZeroAllocations(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping zero-allocation assertion under -race due to runtime map lookup instrumentation")
	}

	tok, err := NewWordPieceTokenizer(filepath.Join("..", "..", "testdata", "tokenizer.json"), "")
	require.NoError(t, err)

	ids := make([]int64, 0, 64)
	offsets := make([]types.Offset, 0, 64)
	input := "User AliceSmith and BobDoe logged in from HTTP API"

	allocs := testing.AllocsPerRun(100, func() {
		ids, offsets, err = tok.Encode(input, ids[:0], offsets[:0])
		if err != nil {
			t.Fatal(err)
		}
	})
	assert.Equal(t, float64(0), allocs, "WordPiece Encode must perform 0 heap allocations with preallocated buffers")
}

func TestWordPieceTokenizer_RuneLengthChangeOffsets(t *testing.T) {
	tok, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tok.Close() }()

	// 'İ' is U+0130 (2 bytes) lowercasing to 'i' (1 byte)
	// 'K' is U+212A (3 bytes) lowercasing to 'k' (1 byte)
	texts := []string{
		"İzmir",
		"Kelvin",
		"İzmir and Kelvin",
	}

	for _, text := range texts {
		ids, offsets, err := tok.Encode(text, nil, nil)
		require.NoError(t, err)
		require.NotEmpty(t, ids)

		for i, off := range offsets {
			assert.GreaterOrEqual(t, off.Start, 0)
			assert.LessOrEqual(t, off.End, len(text))
			assert.LessOrEqual(t, off.Start, off.End)
			if off.Start < len(text) {
				assert.True(t, utf8.RuneStart(text[off.Start]), "offset %d start %d must be rune boundary", i, off.Start)
			}
			if off.End < len(text) {
				assert.True(t, utf8.RuneStart(text[off.End]), "offset %d end %d must be rune boundary", i, off.End)
			}
		}
	}
}

func TestWordPieceTokenizer_FastSlowDifferential(t *testing.T) {
	tokIface, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tokIface.Close() }()

	tok := tokIface.(*wordPieceTokenizer)
	tok.vocab["ⱥ"] = 500
	tok.vocab["##ⱥ"] = 501

	words := []string{
		"alice",
		"Alice",
		"ALICE",
		"AliceSmith",
		"BobDoe",
		"User",
		"HTTP",
		"API",
		"Müller",
		"Zürich",
		"supercalifragilisticexpialidocious",
		"İzmir",
		"Kelvin",
		strings.Repeat("Ⱥ", 10),
		"unknownwordxyz1234567890",
		"a",
		"A",
		"12345",
	}

	for _, w := range words {
		var fastIDs []int64
		var fastOffsets []types.Offset
		fastIDs, fastOffsets = tok.tokenizeWord(w, 0, len(w), fastIDs, fastOffsets)

		var slowIDs []int64
		var slowOffsets []types.Offset
		slowIDs, slowOffsets = tok.tokenizeWordSlow(w, 0, len(w), slowIDs, slowOffsets)

		assert.Equal(t, slowIDs, fastIDs, "word %q: token IDs mismatch between fast and slow path", w)
		assert.Equal(t, slowOffsets, fastOffsets, "word %q: offsets mismatch between fast and slow path", w)
	}
}

func TestWordPieceTokenizer_ExpandingLowercaseRunes(t *testing.T) {
	tokIface, err := NewWordPieceTokenizer(filepath.Clean(validTokenizerPath), validSHA256)
	require.NoError(t, err)
	defer func() { _ = tokIface.Close() }()

	tok := tokIface.(*wordPieceTokenizer)
	// 'Ⱥ' (U+023A, 2 UTF-8 bytes) lowercases to 'ⱥ' (U+2C65, 3 UTF-8 bytes)
	tok.vocab["ⱥ"] = 500
	tok.vocab["##ⱥ"] = 501

	// 100 repetitions of 'Ⱥ': 200 bytes in source text, expands to 300 bytes in lowercase (> 256-byte stack buf)
	word := strings.Repeat("Ⱥ", 100)

	// Case 1: maxInputChar = 256 (passes initial 200-byte rawLen check, overflows 256-byte stack buf,
	// falls back to tokenizeWordSlow, exceeds 256-byte lowercased limit, and emits [UNK] without panicking)
	tok.maxInputChar = 256
	idsUnk, offsetsUnk, err := tok.Encode(word, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []int64{101, 100, 102}, idsUnk)
	assert.Equal(t, []types.Offset{
		{Start: 0, End: 0},
		{Start: 0, End: 200},
		{Start: 200, End: 200},
	}, offsetsUnk)

	// Case 2: maxInputChar = 350 (falls back to tokenizeWordSlow and segments into 100 subwords
	// with exact 2-byte source offsets rather than 3-byte lowercased offsets)
	tok.maxInputChar = 350
	ids, offsets, err := tok.Encode(word, nil, nil)
	require.NoError(t, err)
	require.Len(t, ids, 102)
	assert.Equal(t, int64(500), ids[1])
	assert.Equal(t, int64(501), ids[2])

	for i := 1; i <= 100; i++ {
		expectedStart := (i - 1) * 2
		expectedEnd := i * 2
		assert.Equal(t, types.Offset{Start: expectedStart, End: expectedEnd}, offsets[i], "token %d offset mismatch", i)
		assert.True(t, utf8.RuneStart(word[offsets[i].Start]))
	}
}

func TestWordPieceTokenizer_SlowPathMultiByteRuneLenChange(t *testing.T) {
	tok := &wordPieceTokenizer{
		vocab: map[string]int64{
			"[UNK]": 100, "[CLS]": 101, "[SEP]": 102,
			"i": 300, "##a": 301,
		},
		unkID:        100,
		clsID:        101,
		sepID:        102,
		prefix:       "##",
		maxInputChar: 512,
	}
	// 'İ' (U+0130) is 2 bytes in UTF-8, lowercases to 'i' (1 byte).
	// Repeat 'a' 255 times so prefixLen + rawLen > 256 (triggers tokenizeWordSlow).
	word := "İ" + strings.Repeat("a", 255)
	text := "prefix " + word + " suffix"
	ids, offsets, err := tok.Encode(text, nil, nil)
	require.NoError(t, err)
	require.NotEmpty(t, ids)
	for _, off := range offsets {
		assert.True(t, utf8.ValidString(text[off.Start:off.End]), "offset [%d:%d] must fall on valid UTF-8 rune boundaries", off.Start, off.End)
	}
}

func BenchmarkWordPieceTokenizer_Encode(b *testing.B) {
	tok, err := NewWordPieceTokenizer(filepath.Join("..", "..", "testdata", "tokenizer.json"), "")
	if err != nil {
		b.Fatal(err)
	}

	ids := make([]int64, 0, 64)
	offsets := make([]types.Offset, 0, 64)
	input := "User AliceSmith and BobDoe logged in from HTTP API"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids, offsets, err = tok.Encode(input, ids[:0], offsets[:0])
		if err != nil {
			b.Fatal(err)
		}
	}
}
