package detect

import (
	"context"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// FakeDetector provides a deterministic, in-memory implementation of Detector for testing.
type FakeDetector struct {
	SpansToReturn [][]types.EntitySpan
	ErrToReturn   error
	WarmupCalled  bool
	CloseCalled   bool
}

// NewFakeDetector constructs a new FakeDetector.
func NewFakeDetector() *FakeDetector {
	return &FakeDetector{}
}

func (f *FakeDetector) DetectBatch(ctx context.Context, batch types.BucketedBatch, a *arena.BatchArena) ([][]types.EntitySpan, error) {
	if f.ErrToReturn != nil {
		return nil, f.ErrToReturn
	}
	if len(f.SpansToReturn) > 0 {
		return f.SpansToReturn, nil
	}
	res := make([][]types.EntitySpan, len(batch.Windows))
	return res, nil
}

func (f *FakeDetector) Warmup(ctx context.Context) error {
	f.WarmupCalled = true
	return nil
}

func (f *FakeDetector) Close() error {
	f.CloseCalled = true
	return nil
}
