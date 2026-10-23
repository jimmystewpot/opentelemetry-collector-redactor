package window

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func TestChunker_SliceInput(t *testing.T) {
	c := NewChunker(256, 32, 10)

	// Short string
	prefix, hasTail := c.SliceInput("hello")
	assert.Equal(t, "hello", prefix)
	assert.False(t, hasTail)

	// Exact length
	prefix, hasTail = c.SliceInput("0123456789")
	assert.Equal(t, "0123456789", prefix)
	assert.False(t, hasTail)

	// ASCII oversize
	prefix, hasTail = c.SliceInput("0123456789_extra")
	assert.Equal(t, "0123456789", prefix)
	assert.True(t, hasTail)

	// Multi-byte UTF-8: "日本語" (3 runes, 9 bytes) + "テスト" (3 runes, 9 bytes) = 18 bytes
	// Cut at 10 bytes: 9 bytes is boundary after "日本語". Byte 10 is in the middle of 'テ' (E3 83 86).
	// So cut must fall at 9 bytes ("日本語") without splitting 'テ'.
	text := "日本語テスト"
	prefix, hasTail = c.SliceInput(text)
	assert.Equal(t, "日本語", prefix)
	assert.True(t, hasTail)
	assert.Equal(t, 9, len(prefix))
}

func TestChunker_WindowTokens(t *testing.T) {
	c := NewChunker(256, 32, 16384)

	makeTokens := func(n int) ([]int64, []types.Offset) {
		ids := make([]int64, n)
		offsets := make([]types.Offset, n)
		for i := 0; i < n; i++ {
			ids[i] = int64(100 + i)
			offsets[i] = types.Offset{Start: i * 5, End: (i + 1) * 5}
		}
		return ids, offsets
	}

	// 0 tokens
	var dst []types.TokenWindow
	dst = c.WindowTokens(0, nil, nil, dst)
	assert.Empty(t, dst)

	// 10 tokens -> 1 window
	ids10, off10 := makeTokens(10)
	dst = c.WindowTokens(1, ids10, off10, nil)
	require.Len(t, dst, 1)
	assert.Equal(t, 1, dst[0].ParentUniqueIdx)
	assert.Equal(t, ids10, dst[0].TokenIDs)

	// 256 tokens -> 1 window
	ids256, off256 := makeTokens(256)
	dst = c.WindowTokens(2, ids256, off256, nil)
	require.Len(t, dst, 1)
	assert.Len(t, dst[0].TokenIDs, 256)

	// 257 tokens -> 2 windows: [0..256] and [224..257]
	ids257, off257 := makeTokens(257)
	dst = c.WindowTokens(3, ids257, off257, nil)
	require.Len(t, dst, 2)
	assert.Equal(t, ids257[0:256], dst[0].TokenIDs)
	assert.Equal(t, ids257[224:257], dst[1].TokenIDs)

	// 600 tokens -> 3 windows: [0..256], [224..480], [448..600]
	// Step is 256 - 32 = 224:
	// Window 0: 0..256
	// Window 1: 224..480
	// Window 2: 448..600 (len 152)
	ids600, off600 := makeTokens(600)
	dst = c.WindowTokens(4, ids600, off600, nil)
	require.Len(t, dst, 3)
	assert.Equal(t, ids600[0:256], dst[0].TokenIDs)
	assert.Equal(t, ids600[224:480], dst[1].TokenIDs)
	assert.Equal(t, ids600[448:600], dst[2].TokenIDs)
}

func TestChunker_InvalidOverlapFallback(t *testing.T) {
	// If overlap >= window, step should default to window
	c := NewChunker(100, 100, 0)
	assert.Equal(t, 100, c.step)

	// If maxPayloadBytes <= 0, no truncation
	prefix, hasTail := c.SliceInput(strings.Repeat("A", 1000))
	assert.Len(t, prefix, 1000)
	assert.False(t, hasTail)
}
