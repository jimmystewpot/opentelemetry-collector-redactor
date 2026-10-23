package coalesce

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/collector/pdata/pcommon"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/bucket"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/dedup"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/detect"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/filter"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/window"
)

type dummyTokenizer struct{}

func (d *dummyTokenizer) Encode(text string, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset, error) {
	dstIDs = append(dstIDs, 101, 200, 102)
	dstOffsets = append(dstOffsets, types.Offset{Start: 0, End: 0}, types.Offset{Start: 0, End: len(text)}, types.Offset{Start: len(text), End: len(text)})
	return dstIDs, dstOffsets, nil
}
func (d *dummyTokenizer) Close() error { return nil }

func TestCoalescer_ConcurrentBatchExecution(t *testing.T) {
	pool := arena.NewPool(64, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 64)

	callCount := 0
	var mu sync.Mutex

	fakeDet := detect.NewFakeDetector()
	// Wrap DetectBatch to count actual detector calls
	countingDet := &countingDetector{
		Detector: fakeDet,
		onDetect: func() {
			mu.Lock()
			callCount++
			mu.Unlock()
		},
	}

	c := New(
		50*time.Millisecond, // coalesce window
		16,                  // max strings
		filt,
		tok,
		chunker,
		bucketer,
		countingDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	var wg sync.WaitGroup
	vals := make([]pcommon.Value, 16)

	// 16 concurrent goroutines submitting at the same time
	for i := 0; i < 16; i++ {
		wg.Add(1)
		idx := i
		vals[idx] = pcommon.NewValueStr("duplicate_string_across_goroutines")
		go func() {
			defer wg.Done()
			item := Item{
				Text: vals[idx].Str(),
				Target: dedup.MutationTarget{
					Kind: dedup.TargetPCommonValue,
					Val:  vals[idx],
				},
			}
			err := c.Submit(context.Background(), []Item{item})
			assert.NoError(t, err)
		}()
	}

	wg.Wait()

	mu.Lock()
	actualCalls := callCount
	mu.Unlock()

	// All 16 goroutines should have coalesced into a single detector call!
	assert.Equal(t, 1, actualCalls, "all 16 requests should coalesce into 1 detector call")
}

func TestCoalescer_ContextCancellationBeforeBatch(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	c := New(
		500*time.Millisecond, // long window so it doesn't trigger immediately
		100,                  // high max strings
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	val := pcommon.NewValueStr("item")
	item := Item{
		Text:   val.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val},
	}

	// Cancel context immediately
	cancel()
	err := c.Submit(ctx, []Item{item})
	assert.ErrorIs(t, err, context.Canceled)
}

func TestCoalescer_ContextCancellationDuringInferenceBlocks(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)

	inferenceStarted := make(chan struct{})
	continueInference := make(chan struct{})

	blockingDet := &blockingDetector{
		onStart: func() {
			close(inferenceStarted)
			<-continueInference
		},
	}

	c := New(
		5*time.Millisecond,
		1, // batch size 1 fires immediately
		filt,
		tok,
		chunker,
		bucketer,
		blockingDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	val := pcommon.NewValueStr("val_in_flight")
	item := Item{
		Text:   val.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val},
	}

	done := make(chan error, 1)
	go func() {
		done <- c.Submit(ctx, []Item{item})
	}()

	// Wait until inference starts
	<-inferenceStarted

	// Cancel context while inference is actively running
	cancel()

	// Verify Submit has NOT returned yet (must block until CGo/inference finishes)
	select {
	case <-done:
		t.Fatal("Submit returned before active inference completed!")
	case <-time.After(50 * time.Millisecond):
		// Expected: caller is still blocked waiting for CGo inference
	}

	// Now unblock inference
	close(continueInference)

	// Now Submit should return context.Canceled
	select {
	case err := <-done:
		assert.ErrorIs(t, err, context.Canceled)
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for Submit to complete")
	}
}

func TestCoalescer_ErrorModesAndScrubbing(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	// 1. Invalid UTF-8 under ErrorModeDrop scrubs in-place
	cDrop := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = cDrop.Close() }()

	badVal := pcommon.NewValueStr("bad\xffutf8")
	itemDrop := Item{
		Text:   "bad\xffutf8",
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: badVal},
	}
	err := cDrop.Submit(context.Background(), []Item{itemDrop})
	assert.NoError(t, err)
	assert.Equal(t, types.RedactedUninspected, badVal.Str())

	// 2. Detector error under ErrorModeDrop returns systemic error
	errDet := detect.NewFakeDetector()
	errDet.ErrToReturn = errors.New("inference timeout")
	cErr := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		errDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = cErr.Close() }()

	goodVal := pcommon.NewValueStr("valid_text")
	itemErr := Item{
		Text:   goodVal.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: goodVal},
	}
	err = cErr.Submit(context.Background(), []Item{itemErr})
	assert.ErrorContains(t, err, "inference timeout")

	// 3. Detector error under ErrorModeIgnore passes through with nil error
	cIgnore := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		errDet,
		pool,
		16384,
		types.ErrorModeIgnore,
		false,
	)
	defer func() { _ = cIgnore.Close() }()

	ignoreVal := pcommon.NewValueStr("valid_text")
	itemIgnore := Item{
		Text:   ignoreVal.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: ignoreVal},
	}
	err = cIgnore.Submit(context.Background(), []Item{itemIgnore})
	assert.NoError(t, err)
	assert.Equal(t, "valid_text", ignoreVal.Str(), "should leave text untouched under ignore")
}

func TestCoalescer_DryRun(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()
	// Fake detector returning an entity
	fakeDet.SpansToReturn = [][]types.EntitySpan{
		{
			{Start: 0, End: 4, Label: types.LabelPerson, Confidence: 0.99},
		},
	}

	cDry := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		16384,
		types.ErrorModeDrop,
		true, // DryRun = true
	)
	defer func() { _ = cDry.Close() }()

	val := pcommon.NewValueStr("John Doe")
	item := Item{
		Text:   val.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val},
	}
	err := cDry.Submit(context.Background(), []Item{item})
	assert.NoError(t, err)
	// DryRun should NOT mutate val
	assert.Equal(t, "John Doe", val.Str())
}

type errTokenizer struct{}

func (e *errTokenizer) Encode(text string, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset, error) {
	return nil, nil, errors.New("tokenizer tokenize failed")
}
func (e *errTokenizer) Close() error { return nil }

func TestCoalescer_TokenizerError(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &errTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	c := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	val := pcommon.NewValueStr("some_string")
	item := Item{
		Text:   val.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val},
	}
	err := c.Submit(context.Background(), []Item{item})
	assert.NoError(t, err)
	assert.Equal(t, types.RedactedUninspected, val.Str())
}

func TestCoalescer_OversizeTail(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	// Payload cap = 10 bytes
	chunker := window.NewChunker(32, 4, 10)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	c := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		10,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	oversizeText := "0123456789extra_tail"
	val := pcommon.NewValueStr(oversizeText)
	item := Item{
		Text:   oversizeText,
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val},
	}
	err := c.Submit(context.Background(), []Item{item})
	assert.NoError(t, err)
	assert.Equal(t, "0123456789"+types.RedactedUninspected, val.Str())
}

type testRecorder struct {
	batches int
}

func (tr *testRecorder) RecordBatch(coalescedStrings int)          { tr.batches++ }
func (tr *testRecorder) RecordTensorBatch(int, float64)            {}
func (tr *testRecorder) RecordStageDuration(string, time.Duration) {}
func (tr *testRecorder) RecordEntity(types.Label)                  {}
func (tr *testRecorder) RecordScrubbed(string, string)             {}
func (tr *testRecorder) RecordError(string)                        {}
func (tr *testRecorder) RecordSkipped(string, int)                 {}
func (tr *testRecorder) RecordDropped(string, string, int)         {}
func (tr *testRecorder) RecordInflight(int64)                      {}
func (tr *testRecorder) RecordQueueWait(time.Duration)             {}
func (tr *testRecorder) RecordMetricCollision(string, int)         {}
func (tr *testRecorder) RecordMetricLabelPII(string, int)          {}

func TestCoalescer_OptionsAndClose(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()
	rec := &testRecorder{}

	c := New(
		500*time.Millisecond,
		100,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
		WithTimeout(10*time.Millisecond),
		WithTelemetry(rec),
	)

	// Empty items
	assert.NoError(t, c.Submit(context.Background(), nil))

	// Submit item then close coalescer immediately
	val := pcommon.NewValueStr("pending_item")
	item := Item{
		Text:   val.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val},
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- c.Submit(context.Background(), []Item{item})
	}()

	time.Sleep(10 * time.Millisecond)
	_ = c.Close()
	// Multiple close is safe
	_ = c.Close()

	err := <-errCh
	assert.ErrorContains(t, err, "coalescer closed")

	// Submit after close
	errAfter := c.Submit(context.Background(), []Item{item})
	assert.ErrorContains(t, errAfter, "coalescer is closed")
}

func TestCoalescer_TimerTrigger(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	// 10ms window, high maxStrings (50) -> must trigger via timer
	c := New(
		10*time.Millisecond,
		50,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	val := pcommon.NewValueStr("timer_triggered_string")
	item := Item{
		Text:   val.Str(),
		Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val},
	}

	start := time.Now()
	err := c.Submit(context.Background(), []Item{item})
	assert.NoError(t, err)
	assert.True(t, time.Since(start) >= 10*time.Millisecond, "should have waited for coalesce window timer")
}

func TestCoalescer_DefaultLimits(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	// 0 limits should be set to defaults
	c := New(
		0,
		0,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		0,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	assert.Equal(t, 2*time.Millisecond, c.window)
	assert.Equal(t, 64, c.maxStrings)
	assert.Equal(t, 16384, c.maxPayloadBytes)

	// Pre-canceled context
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	errPre := c.Submit(canceledCtx, []Item{{Text: "pre"}})
	assert.ErrorIs(t, errPre, context.Canceled)
}

func TestCoalescer_MultiplePendingCancellation(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)
	fakeDet := detect.NewFakeDetector()

	c := New(
		200*time.Millisecond,
		50,
		filt,
		tok,
		chunker,
		bucketer,
		fakeDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	ctx1, cancel1 := context.WithCancel(context.Background())
	val1 := pcommon.NewValueStr("item1")
	val2 := pcommon.NewValueStr("item2")

	// Submit first item (starts timer)
	done1 := make(chan error, 1)
	go func() {
		done1 <- c.Submit(ctx1, []Item{{Text: val1.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val1}}})
	}()

	// Wait brief moment so req1 is pending
	time.Sleep(10 * time.Millisecond)

	// Submit second item while timer is active
	done2 := make(chan error, 1)
	go func() {
		done2 <- c.Submit(context.Background(), []Item{{Text: val2.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val2}}})
	}()

	time.Sleep(10 * time.Millisecond)

	// Cancel req1 while req2 remains pending
	cancel1()

	err1 := <-done1
	assert.ErrorIs(t, err1, context.Canceled)

	// req2 should eventually succeed when timer fires
	err2 := <-done2
	assert.NoError(t, err2)
}

type panicDetector struct{}

func (p *panicDetector) DetectBatch(_ context.Context, _ types.BucketedBatch, _ *arena.BatchArena) ([][]types.EntitySpan, error) {
	panic("unexpected native crash in detector")
}
func (p *panicDetector) Warmup(_ context.Context) error { return nil }
func (p *panicDetector) Close() error                   { return nil }

func TestCoalescer_PanicRecovery(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)

	c := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		&panicDetector{},
		pool,
		16384,
		types.ErrorModeDrop,
		false,
	)
	defer func() { _ = c.Close() }()

	val := pcommon.NewValueStr("test text")
	item := Item{Text: val.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val}}

	err := c.Submit(context.Background(), []Item{item})
	assert.ErrorContains(t, err, "redaction batch panic")
}

func TestCoalescer_SemaphoreFull(t *testing.T) {
	pool := arena.NewPool(16, 32, 19)
	filt := filter.New(4, nil)
	tok := &dummyTokenizer{}
	chunker := window.NewChunker(32, 4, 16384)
	bucketer := bucket.New([]int{32}, 16)

	inFlight := make(chan struct{})
	unblock := make(chan struct{})

	blockingDet := &blockingDetector{
		onStart: func() {
			close(inFlight)
			<-unblock
		},
	}

	// Sized with 1 concurrent run
	c := New(
		5*time.Millisecond,
		1,
		filt,
		tok,
		chunker,
		bucketer,
		blockingDet,
		pool,
		16384,
		types.ErrorModeDrop,
		false,
		WithConcurrentRuns(1),
	)
	defer func() { _ = c.Close() }()

	val1 := pcommon.NewValueStr("first")
	val2 := pcommon.NewValueStr("second")

	go func() {
		_ = c.Submit(context.Background(), []Item{{Text: val1.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val1}}})
	}()

	<-inFlight

	// Second request arrives while semaphore slot is occupied
	err := c.Submit(context.Background(), []Item{{Text: val2.Str(), Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val2}}})
	assert.ErrorContains(t, err, "semaphore full")

	close(unblock)
}

type countingDetector struct {
	detect.Detector
	onDetect func()
}

func (cd *countingDetector) DetectBatch(ctx context.Context, batch types.BucketedBatch, a *arena.BatchArena) ([][]types.EntitySpan, error) {
	if cd.onDetect != nil {
		cd.onDetect()
	}
	return cd.Detector.DetectBatch(ctx, batch, a)
}

type blockingDetector struct {
	onStart func()
}

func (bd *blockingDetector) DetectBatch(ctx context.Context, batch types.BucketedBatch, a *arena.BatchArena) ([][]types.EntitySpan, error) {
	if bd.onStart != nil {
		bd.onStart()
	}
	res := make([][]types.EntitySpan, len(batch.Windows))
	return res, nil
}
func (bd *blockingDetector) Warmup(ctx context.Context) error { return nil }
func (bd *blockingDetector) Close() error                     { return nil }
