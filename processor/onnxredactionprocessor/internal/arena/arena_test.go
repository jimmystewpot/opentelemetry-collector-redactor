package arena

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func TestPool_ZeroingOnRelease(t *testing.T) {
	pool := NewPool(32, 256, 19)
	a := pool.Acquire()
	require.NotNil(t, a)

	// Dirty the arena slices
	a.InputIDs[0] = 12345
	a.AttentionMask[0] = 1
	a.Logits[0] = 99.9
	a.TokenIDPool = append(a.TokenIDPool, 54321)
	a.OffsetPool = append(a.OffsetPool, types.Offset{Start: 0, End: 5})
	a.Windows = append(a.Windows, types.TokenWindow{ParentUniqueIdx: 1})
	a.ScratchBytes = append(a.ScratchBytes, "sensitive_data"...)

	pool.Release(a)

	// Reacquire and verify all buffers are zeroed / reset
	b := pool.Acquire()
	defer pool.Release(b)

	assert.Equal(t, int64(0), b.InputIDs[0], "InputIDs must be zeroed")
	assert.Equal(t, int64(0), b.AttentionMask[0], "AttentionMask must be zeroed")
	assert.Equal(t, float32(0), b.Logits[0], "Logits must be zeroed")
	assert.Empty(t, b.TokenIDPool, "TokenIDPool must be reset to len 0")
	assert.Empty(t, b.OffsetPool, "OffsetPool must be reset to len 0")
	assert.Empty(t, b.Windows, "Windows must be reset to len 0")
	assert.Empty(t, b.ScratchBytes, "ScratchBytes must be reset to len 0")
	assert.Empty(t, b.Dedup.UniqueStrings(), "Dedup must be reset")
}

func TestPool_ZeroAllocsPerRun(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping zero-allocation assertion under -race due to runtime shadow memory instrumentation")
	}

	pool := NewPool(32, 256, 19)

	// Warm up pool
	warm := pool.Acquire()
	pool.Release(warm)

	var a *BatchArena
	allocs := testing.AllocsPerRun(100, func() {
		a = pool.Acquire()
		pool.Release(a)
	})

	assert.Equal(t, float64(0), allocs, "pool Acquire/Release must be 0 allocs/op")

	// Release nil safety
	pool.Release(nil)

	// Defaults fallback
	defaultPool := NewPool(0, 0, 0)
	dArena := defaultPool.Acquire()
	require.NotNil(t, dArena)
	defaultPool.Release(dArena)
}

func TestPool_OversizedSliceCapReset(t *testing.T) {
	pool := NewPool(4, 16, 19)
	a := pool.Acquire()

	// Dirty index 30 (simulating Sub-batch 1), then slice down to [:8] (simulating Sub-batch 2)
	a.InputIDs[30] = 999
	a.AttentionMask[30] = 1
	a.Logits[30] = 42.0
	a.InputIDs = a.InputIDs[:8]
	a.AttentionMask = a.AttentionMask[:8]
	a.Logits = a.Logits[:8*19]

	// Grow TokenIDPool, OffsetPool, Windows, SpanScratch, and ScratchBytes past 4x baseline
	a.TokenIDPool = make([]int64, 1000)
	a.OffsetPool = make([]types.Offset, 1000)
	a.Windows = make([]types.TokenWindow, 100)
	a.SpanScratch = make([]types.EntitySpan, 500)
	a.ScratchBytes = make([]byte, 131072)

	pool.Release(a)

	b := pool.Acquire()
	defer pool.Release(b)

	assert.Equal(t, 64, len(b.InputIDs), "InputIDs length must be restored to tensorElements")
	assert.Equal(t, 64*19, len(b.Logits), "Logits length must be restored to logitsElements")
	assert.Equal(t, int64(0), b.InputIDs[30], "InputIDs beyond final sub-batch slice must still be zeroed")
	assert.Equal(t, int64(0), b.AttentionMask[30], "AttentionMask beyond final sub-batch slice must still be zeroed")
	assert.Equal(t, float32(0), b.Logits[30], "Logits beyond final sub-batch slice must still be zeroed")
	assert.LessOrEqual(t, cap(b.TokenIDPool), 256, "oversized TokenIDPool must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.OffsetPool), 256, "oversized OffsetPool must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.Windows), 16, "oversized Windows must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.SpanScratch), 256, "oversized SpanScratch must be reallocated to baseline")
	assert.LessOrEqual(t, cap(b.ScratchBytes), 65536, "oversized ScratchBytes must be reallocated to baseline")
}
