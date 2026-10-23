package detect

import (
	"context"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// Detector defines the inference interface for evaluating telemetry token windows
// and emitting entity spans.
type Detector interface {
	DetectBatch(ctx context.Context, batch types.BucketedBatch, a *arena.BatchArena) ([][]types.EntitySpan, error)
	Warmup(ctx context.Context) error
	Close() error
}
