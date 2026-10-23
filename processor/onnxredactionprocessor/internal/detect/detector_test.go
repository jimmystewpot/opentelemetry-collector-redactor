package detect

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func TestFakeDetector(t *testing.T) {
	fd := NewFakeDetector()
	require.NoError(t, fd.Warmup(context.Background()))
	assert.True(t, fd.WarmupCalled)

	w := types.TokenWindow{ParentUniqueIdx: 0}
	batch := types.BucketedBatch{Windows: []types.TokenWindow{w}}

	spans, err := fd.DetectBatch(context.Background(), batch, nil)
	require.NoError(t, err)
	require.Len(t, spans, 1)
	assert.Empty(t, spans[0])

	// Custom return
	fd.SpansToReturn = [][]types.EntitySpan{{{Start: 0, End: 5, Label: types.LabelPerson}}}
	spans2, err := fd.DetectBatch(context.Background(), batch, nil)
	require.NoError(t, err)
	require.Len(t, spans2, 1)
	assert.Equal(t, types.LabelPerson, spans2[0][0].Label)

	// Error return
	expectedErr := errors.New("simulated error")
	fd.ErrToReturn = expectedErr
	_, err = fd.DetectBatch(context.Background(), batch, nil)
	require.Equal(t, expectedErr, err)

	require.NoError(t, fd.Close())
	assert.True(t, fd.CloseCalled)
}
