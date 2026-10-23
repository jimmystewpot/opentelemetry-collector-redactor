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
