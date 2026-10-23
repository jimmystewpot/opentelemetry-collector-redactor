package bucket

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func makeWindow(parentIdx, seqLen int) types.TokenWindow {
	ids := make([]int64, seqLen)
	return types.TokenWindow{
		ParentUniqueIdx: parentIdx,
		TokenIDs:        ids,
	}
}

func TestBucketer_TightPadding(t *testing.T) {
	b := New([]int{32, 128, 256}, 32)

	// Windows of length 50, 60, 70 tokens (all fall into the <= 128 bucket)
	windows := []types.TokenWindow{
		makeWindow(1, 50),
		makeWindow(2, 60),
		makeWindow(3, 70),
	}

	batches := b.Group(windows)
	require.Len(t, batches, 1)

	// MaxSeqLen must be tightly padded to 70 (not 128!)
	assert.Equal(t, 70, batches[0].MaxSeqLen)
	assert.Len(t, batches[0].Windows, 3)
}

func TestBucketer_MultiBucketAndBatchSplitting(t *testing.T) {
	b := New([]int{32, 128, 256}, 32)

	var windows []types.TokenWindow

	// 5 windows of length 20 (bucket <= 32)
	for i := 0; i < 5; i++ {
		windows = append(windows, makeWindow(i, 20))
	}

	// 40 windows of length 100 (bucket <= 128, should split into two sub-batches of 32 and 8)
	for i := 0; i < 40; i++ {
		windows = append(windows, makeWindow(100+i, 100))
	}

	// 2 windows of length 200 (bucket <= 256)
	windows = append(windows, makeWindow(200, 180), makeWindow(201, 200))

	batches := b.Group(windows)
	require.Len(t, batches, 4)

	// Batch 0: <= 32 bucket (5 windows, MaxSeqLen = 20)
	assert.Equal(t, 20, batches[0].MaxSeqLen)
	assert.Len(t, batches[0].Windows, 5)

	// Batch 1: first sub-batch of <= 128 bucket (32 windows, MaxSeqLen = 100)
	assert.Equal(t, 100, batches[1].MaxSeqLen)
	assert.Len(t, batches[1].Windows, 32)

	// Batch 2: second sub-batch of <= 128 bucket (8 windows, MaxSeqLen = 100)
	assert.Equal(t, 100, batches[2].MaxSeqLen)
	assert.Len(t, batches[2].Windows, 8)

	// Batch 3: <= 256 bucket (2 windows, MaxSeqLen = 200)
	assert.Equal(t, 200, batches[3].MaxSeqLen)
	assert.Len(t, batches[3].Windows, 2)
}

func TestBucketer_EmptyAndDefaults(t *testing.T) {
	b := New(nil, 0) // should fallback to defaults
	assert.Empty(t, b.Group(nil))

	w := makeWindow(1, 10)
	batches := b.Group([]types.TokenWindow{w})
	require.Len(t, batches, 1)
	assert.Equal(t, 10, batches[0].MaxSeqLen)
}
