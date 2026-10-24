package ort

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/decode"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

const (
	tinyModelPath = "../../testdata/tiny_ner.onnx"
	tinySHA       = "7ef06dbfe51de3cd4cb08c2075a390b21ad261b28e7c7f9e36e6e66d2d7329a8"

	slowModelPath = "../../testdata/slow_loop.onnx"
	slowSHA       = "8cdd11a6501b9185db54f291bb74378ce3213594baa53d9c7d2e3af6540c3b8b"
)

var defaultBIO = []string{
	"O",
	"B-PERSON", "B-EMAIL", "B-PHONE", "B-LOCATION", "B-CREDIT_CARD", "B-SSN", "B-IBAN", "B-MEDICAL_ID", "B-PATH_USER",
	"I-PERSON", "I-EMAIL", "I-PHONE", "I-LOCATION", "I-CREDIT_CARD", "I-SSN", "I-IBAN", "I-MEDICAL_ID", "I-PATH_USER",
}

func TestEngine_ChecksumVerification(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)

	// Valid SHA
	eng, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 1, false, dec)
	require.NoError(t, err)
	require.NotNil(t, eng)
	defer func() { _ = eng.Close() }()

	// Mismatched SHA
	_, err = NewEngine(filepath.Clean(tinyModelPath), "deadbeef00000000000000000000000000000000000000000000000000000000", 1, 1, false, dec)
	require.Error(t, err, "must reject mismatched SHA-256")
}

func TestEngine_InferenceTinyNER(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 1, false, dec)
	require.NoError(t, err)
	defer func() { _ = eng.Close() }()

	require.NoError(t, eng.Warmup(context.Background()))

	pool := arena.NewPool(4, 16, 19)
	a := pool.Acquire()
	defer pool.Release(a)

	// Window with token 200 (B-PERSON) and 210 (I-PERSON)
	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 210, 102},
		Offsets: []types.Offset{
			{Start: 0, End: 0},
			{Start: 0, End: 4},
			{Start: 5, End: 8},
			{Start: 8, End: 8},
		},
	}

	batch := types.BucketedBatch{
		MaxSeqLen: 4,
		Windows:   []types.TokenWindow{w},
	}

	spans, err := eng.DetectBatch(context.Background(), batch, a)
	require.NoError(t, err)
	require.Len(t, spans, 1)

	// The tiny_ner model deterministic mapping maps 200 -> B-PERSON and 210 -> I-PERSON
	require.NotEmpty(t, spans[0])
	assert.Equal(t, 0, spans[0][0].Start)
	assert.Equal(t, 8, spans[0][0].End)
	assert.Equal(t, types.LabelPerson, spans[0][0].Label)
}

func TestEngine_CooperativeCancellationSlowLoop(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(slowModelPath), slowSHA, 1, 1, false, dec)
	require.NoError(t, err)
	defer func() { _ = eng.Close() }()

	pool := arena.NewPool(2, 8, 19)
	a := pool.Acquire()
	defer pool.Release(a)

	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 102},
		Offsets: []types.Offset{
			{Start: 0, End: 0},
			{Start: 0, End: 4},
			{Start: 4, End: 4},
		},
	}
	batch := types.BucketedBatch{
		MaxSeqLen: 3,
		Windows:   []types.TokenWindow{w},
	}

	// Run with 5ms timeout: slow_loop has 500,000 iterations and will be terminated
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()

	_, err = eng.DetectBatch(ctx, batch, a)
	require.Error(t, err, "slow loop should be aborted via cooperative cancellation")

	// Verify engine is still functional after cancellation (terminate unset)
	engTiny, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 1, false, dec)
	require.NoError(t, err)
	defer func() { _ = engTiny.Close() }()

	spans, err := engTiny.DetectBatch(context.Background(), batch, a)
	require.NoError(t, err)
	assert.NotNil(t, spans)
}

func TestEngine_ConcurrentCancellationIsolation(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 2, false, dec)
	require.NoError(t, err)
	defer func() { _ = eng.Close() }()

	pool := arena.NewPool(4, 16, 19)

	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 102},
		Offsets: []types.Offset{
			{Start: 0, End: 0},
			{Start: 0, End: 4},
			{Start: 4, End: 4},
		},
	}
	batch := types.BucketedBatch{
		MaxSeqLen: 3,
		Windows:   []types.TokenWindow{w},
	}

	// Launch a canceled run and a normal run concurrently
	ctxCanceled, cancel := context.WithCancel(context.Background())
	cancel() // canceled before/at start

	doneCh := make(chan error, 2)
	go func() {
		a1 := pool.Acquire()
		defer pool.Release(a1)
		_, errRun := eng.DetectBatch(ctxCanceled, batch, a1)
		doneCh <- errRun
	}()

	go func() {
		a2 := pool.Acquire()
		defer pool.Release(a2)
		_, errRun := eng.DetectBatch(context.Background(), batch, a2)
		doneCh <- errRun
	}()

	err1 := <-doneCh
	err2 := <-doneCh

	// One should fail with context.Canceled, the other must succeed!
	if err1 != nil {
		assert.ErrorIs(t, err1, context.Canceled)
		assert.NoError(t, err2, "concurrent run must NOT be affected by another run's cancellation")
	} else {
		assert.NoError(t, err1, "concurrent run must NOT be affected by another run's cancellation")
		assert.ErrorIs(t, err2, context.Canceled)
	}
}

func TestEngine_ConcurrentCloseDuringInference(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(slowModelPath), slowSHA, 1, 2, false, dec)
	require.NoError(t, err)

	pool := arena.NewPool(2, 8, 19)
	a := pool.Acquire()
	defer pool.Release(a)

	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 102},
		Offsets:         []types.Offset{{Start: 0, End: 0}, {Start: 0, End: 4}, {Start: 4, End: 4}},
	}
	batch := types.BucketedBatch{MaxSeqLen: 3, Windows: []types.TokenWindow{w}}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = eng.DetectBatch(ctx, batch, a)
	}()

	time.Sleep(2 * time.Millisecond)
	require.NoError(t, eng.Close())
	<-done

	_, errAfter := eng.DetectBatch(context.Background(), batch, a)
	require.ErrorContains(t, errAfter, "session is closed")
}

func TestEngine_ConcurrentNewEngineInit(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	const concurrent = 8
	errCh := make(chan error, concurrent)
	engines := make([]*Engine, concurrent)

	var wg sync.WaitGroup
	for i := 0; i < concurrent; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			eng, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 1, false, dec)
			if err != nil {
				errCh <- err
				return
			}
			engines[idx] = eng
		}(i)
	}
	wg.Wait()
	close(errCh)

	for err := range errCh {
		require.NoError(t, err)
	}
	for _, eng := range engines {
		if eng != nil {
			_ = eng.Close()
		}
	}
}

func TestEngine_SpanScratchReuse(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 1, false, dec)
	require.NoError(t, err)
	defer func() { _ = eng.Close() }()

	pool := arena.NewPool(4, 16, 19)
	a := pool.Acquire()
	defer pool.Release(a)

	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 210, 102},
		Offsets: []types.Offset{
			{Start: 0, End: 0},
			{Start: 0, End: 4},
			{Start: 5, End: 8},
			{Start: 8, End: 8},
		},
	}
	batch := types.BucketedBatch{
		MaxSeqLen: 4,
		Windows:   []types.TokenWindow{w, w},
	}

	spans, err := eng.DetectBatch(context.Background(), batch, a)
	require.NoError(t, err)
	require.Len(t, spans, 2)
	assert.NotEmpty(t, spans[0])
	assert.NotEmpty(t, spans[1])
	assert.NotEmpty(t, a.SpanScratch, "SpanScratch must be populated by DetectBatch")
}

func TestEngine_ZeroLengthLogitsSliceAndZeroMaxSeqLen(t *testing.T) {
	dec := decode.NewDecoder(defaultBIO, nil, 0.85)
	eng, err := NewEngine(filepath.Clean(tinyModelPath), tinySHA, 1, 1, false, dec)
	require.NoError(t, err)
	defer func() { _ = eng.Close() }()

	pool := arena.NewPool(4, 16, 19)
	a := pool.Acquire()
	defer pool.Release(a)

	// 1. MaxSeqLen == 0 must return nil, nil without panicking on &a.InputIDs[0]
	emptySeqBatch := types.BucketedBatch{
		MaxSeqLen: 0,
		Windows:   []types.TokenWindow{{ParentUniqueIdx: 0}},
	}
	spans, err := eng.DetectBatch(context.Background(), emptySeqBatch, a)
	require.NoError(t, err)
	assert.Nil(t, spans)

	// 2. Arena with len(a.Logits) == 0 but cap(a.Logits) > 0 must reslice and succeed
	a.Logits = a.Logits[:0]
	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 200, 102},
		Offsets:         []types.Offset{{Start: 0, End: 0}, {Start: 0, End: 4}, {Start: 4, End: 4}},
	}
	validBatch := types.BucketedBatch{MaxSeqLen: 3, Windows: []types.TokenWindow{w}}
	spans, err = eng.DetectBatch(context.Background(), validBatch, a)
	require.NoError(t, err)
	require.Len(t, spans, 1)
}
