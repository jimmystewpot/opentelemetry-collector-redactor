package arena

import (
	"sync"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/dedup"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// BatchArena holds preallocated, reusable slices for a batch inference pipeline,
// eliminating heap allocations across tokenization, windowing, deduplication, and CGo tensors.
type BatchArena struct {
	Dedup         *dedup.Table
	Windows       []types.TokenWindow
	TokenIDPool   []int64
	OffsetPool    []types.Offset
	InputIDs      []int64
	AttentionMask []int64
	Logits        []float32
	SpanScratch   []types.EntitySpan
	ScratchBytes  []byte
}

// Pool manages a pool of BatchArenas via sync.Pool.
type Pool struct {
	pool sync.Pool
}

// NewPool initializes a BatchArena pool dimensioned by maxBatchSize, maxSeqLen, and numClasses.
func NewPool(maxBatchSize, maxSeqLen, numClasses int) *Pool {
	if maxBatchSize <= 0 {
		maxBatchSize = 32
	}
	if maxSeqLen <= 0 {
		maxSeqLen = 256
	}
	if numClasses <= 0 {
		numClasses = 19
	}

	tensorElements := maxBatchSize * maxSeqLen
	logitsElements := tensorElements * numClasses

	p := &Pool{}
	p.pool.New = func() any {
		return &BatchArena{
			Dedup:         dedup.NewTable(maxBatchSize * 4),
			Windows:       make([]types.TokenWindow, 0, maxBatchSize),
			TokenIDPool:   make([]int64, 0, tensorElements),
			OffsetPool:    make([]types.Offset, 0, tensorElements),
			InputIDs:      make([]int64, tensorElements),
			AttentionMask: make([]int64, tensorElements),
			Logits:        make([]float32, logitsElements),
			SpanScratch:   make([]types.EntitySpan, 0, 64),
			ScratchBytes:  make([]byte, 0, 16384),
		}
	}
	return p
}

// Acquire retrieves a BatchArena from the pool.
func (p *Pool) Acquire() *BatchArena {
	return p.pool.Get().(*BatchArena)
}

// Release zeroes all sensitive tensor and token memory and returns the arena to the pool.
func (p *Pool) Release(a *BatchArena) {
	if a == nil {
		return
	}

	// Security zeroing of sensitive buffers
	clear(a.InputIDs)
	clear(a.AttentionMask)
	clear(a.Logits)

	if cap(a.TokenIDPool) > 0 {
		clear(a.TokenIDPool[:cap(a.TokenIDPool)])
		a.TokenIDPool = a.TokenIDPool[:0]
	}
	a.OffsetPool = a.OffsetPool[:0]
	a.Windows = a.Windows[:0]
	a.SpanScratch = a.SpanScratch[:0]

	if cap(a.ScratchBytes) > 0 {
		clear(a.ScratchBytes[:cap(a.ScratchBytes)])
		a.ScratchBytes = a.ScratchBytes[:0]
	}

	a.Dedup.Reset()

	p.pool.Put(a)
}
