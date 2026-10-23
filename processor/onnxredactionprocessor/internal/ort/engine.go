package ort

/*
#cgo CFLAGS: -I${SRCDIR}/../../../../.ort/include
#cgo LDFLAGS: -L${SRCDIR}/../../../../.ort/lib -lonnxruntime -Wl,-rpath,${SRCDIR}/../../../../.ort/lib
#include "ort_shim.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unsafe"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/decode"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/detect"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// Engine manages an ONNX Runtime inference session via a zero-copy CGo bridge.
type Engine struct {
	modelPath   string
	env         unsafe.Pointer
	session     unsafe.Pointer
	runOptsPool chan unsafe.Pointer
	decoder     *decode.Decoder
	warmupArena *arena.BatchArena
	mu          sync.RWMutex
}

var _ detect.Detector = (*Engine)(nil)

var (
	initOnce sync.Once
	initErr  error
)

// NewEngine creates and initializes an ONNX Runtime inference session with checksum validation.
func NewEngine(modelPath, expectedSHA string, intraOpThreads, concurrentRuns int, allowSpinning bool, dec *decode.Decoder) (*Engine, error) {
	cleanPath := filepath.Clean(modelPath)
	data, err := os.ReadFile(cleanPath) //nolint:gosec // Model path configured via operator config
	if err != nil {
		return nil, fmt.Errorf("ort: failed to read model %s: %w", cleanPath, err)
	}

	if expectedSHA != "" {
		h := sha256.Sum256(data)
		actualSHA := hex.EncodeToString(h[:])
		if !strings.EqualFold(actualSHA, expectedSHA) {
			return nil, fmt.Errorf("ort: checksum mismatch for model %s: expected %s, got %s",
				cleanPath, expectedSHA, actualSHA)
		}
	}

	initOnce.Do(func() {
		var cErr *C.char
		rc := C.ort_shim_init(&cErr)
		if rc != 0 {
			errMsg := "unknown error"
			if cErr != nil {
				errMsg = C.GoString(cErr)
				C.ort_shim_free_string(cErr)
			}
			initErr = fmt.Errorf("ort: failed to initialize runtime: %s", errMsg)
		}
	})
	if initErr != nil {
		return nil, initErr
	}

	cPath := C.CString(cleanPath)
	defer C.free(unsafe.Pointer(cPath))

	var cEnv, cSession, cRunOpts unsafe.Pointer
	var cErr *C.char

	spinning := 0
	if allowSpinning {
		spinning = 1
	}

	rc := C.ort_shim_create_session(
		cPath,
		C.int(intraOpThreads),
		C.int(concurrentRuns),
		C.int(spinning),
		&cEnv,
		&cSession,
		&cRunOpts,
		&cErr,
	)
	if rc != 0 {
		errMsg := "unknown error"
		if cErr != nil {
			errMsg = C.GoString(cErr)
			C.ort_shim_free_string(cErr)
		}
		return nil, fmt.Errorf("ort: failed to create session: %s", errMsg)
	}

	if concurrentRuns <= 0 {
		concurrentRuns = 2
	}
	runOptsPool := make(chan unsafe.Pointer, concurrentRuns)
	if cRunOpts != nil {
		runOptsPool <- cRunOpts
	}
	for i := 1; i < concurrentRuns; i++ {
		var opt unsafe.Pointer
		rcOpt := C.ort_shim_create_run_options(&opt, &cErr)
		if rcOpt != 0 {
			close(runOptsPool)
			for o := range runOptsPool {
				C.ort_shim_release_run_options(o)
			}
			C.ort_shim_destroy_session(cEnv, cSession, nil)
			errMsg := "unknown error"
			if cErr != nil {
				errMsg = C.GoString(cErr)
				C.ort_shim_free_string(cErr)
			}
			return nil, fmt.Errorf("ort: failed to create run options: %s", errMsg)
		}
		runOptsPool <- opt
	}

	warmArena := arena.NewPool(1, 2, 19).Acquire()

	return &Engine{
		modelPath:   cleanPath,
		env:         cEnv,
		session:     cSession,
		runOptsPool: runOptsPool,
		decoder:     dec,
		warmupArena: warmArena,
	}, nil
}

// Close destroys the underlying ORT session and run options.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.session != nil {
		C.ort_shim_destroy_session(e.env, e.session, nil)
		e.session = nil
		e.env = nil
	}
	if e.runOptsPool != nil {
		close(e.runOptsPool)
		for opt := range e.runOptsPool {
			C.ort_shim_release_run_options(opt)
		}
		e.runOptsPool = nil
	}
	return nil
}

// Warmup runs a minimal dummy inference pass through the model to trigger execution provider initialization.
func (e *Engine) Warmup(ctx context.Context) error {
	w := types.TokenWindow{
		ParentUniqueIdx: 0,
		TokenIDs:        []int64{101, 102},
		Offsets:         []types.Offset{{Start: 0, End: 0}, {Start: 0, End: 0}},
	}
	batch := types.BucketedBatch{
		MaxSeqLen: 2,
		Windows:   []types.TokenWindow{w},
	}
	_, err := e.DetectBatch(ctx, batch, e.warmupArena)
	return err
}

// DetectBatch executes native ONNX inference over a bucketed batch and returns decoded EntitySpans.
func (e *Engine) DetectBatch(ctx context.Context, batch types.BucketedBatch, a *arena.BatchArena) ([][]types.EntitySpan, error) {
	if len(batch.Windows) == 0 {
		return nil, nil
	}
	e.mu.RLock()
	if e.session == nil {
		e.mu.RUnlock()
		return nil, errors.New("ort: session is closed")
	}

	var runOpts unsafe.Pointer
	select {
	case runOpts = <-e.runOptsPool:
	default:
		var cErr *C.char
		rc := C.ort_shim_create_run_options(&runOpts, &cErr)
		if rc != 0 {
			e.mu.RUnlock()
			errMsg := "unknown error"
			if cErr != nil {
				errMsg = C.GoString(cErr)
				C.ort_shim_free_string(cErr)
			}
			return nil, fmt.Errorf("ort: failed to create run options: %s", errMsg)
		}
	}
	defer func() {
		C.ort_shim_unset_terminate(runOpts)
		select {
		case e.runOptsPool <- runOpts:
		default:
			C.ort_shim_release_run_options(runOpts)
		}
		e.mu.RUnlock()
	}()

	batchSize := len(batch.Windows)
	seqLen := batch.MaxSeqLen
	totalElements := batchSize * seqLen

	if cap(a.InputIDs) < totalElements || cap(a.AttentionMask) < totalElements {
		return nil, fmt.Errorf("ort: arena capacity %d is insufficient for batchSize=%d, seqLen=%d (%d)",
			cap(a.InputIDs), batchSize, seqLen, totalElements)
	}

	a.InputIDs = a.InputIDs[:totalElements]
	a.AttentionMask = a.AttentionMask[:totalElements]
	clear(a.InputIDs)
	clear(a.AttentionMask)

	// Populate flattened tensors with padding
	for i, w := range batch.Windows {
		offset := i * seqLen
		wLen := len(w.TokenIDs)
		for j := 0; j < wLen && j < seqLen; j++ {
			a.InputIDs[offset+j] = w.TokenIDs[j]
			a.AttentionMask[offset+j] = 1
		}
	}

	// Cooperative cancellation monitor
	var done, monitorDone chan struct{}
	if ctx.Done() != nil {
		done = make(chan struct{})
		monitorDone = make(chan struct{})

		go func() {
			defer close(monitorDone)
			select {
			case <-ctx.Done():
				C.ort_shim_terminate(runOpts)
			case <-done:
			}
		}()
	}

	var cErr *C.char
	var outNumClasses C.int64_t = 19
	outLogitsCap := C.int64_t(cap(a.Logits))

	inIDsPtr := (*C.int64_t)(unsafe.Pointer(&a.InputIDs[0]))
	attnMaskPtr := (*C.int64_t)(unsafe.Pointer(&a.AttentionMask[0]))
	logitsPtr := (*C.float)(unsafe.Pointer(&a.Logits[0]))

	rc := C.ort_shim_run_batch(
		e.session,
		runOpts,
		inIDsPtr,
		attnMaskPtr,
		C.int64_t(batchSize),
		C.int64_t(seqLen),
		logitsPtr,
		outLogitsCap,
		&outNumClasses,
		&cErr,
	)

	// Ensure monitor goroutine has exited before unsetting terminate flag
	if monitorDone != nil {
		close(done)
		<-monitorDone
	}

	if ctx.Err() != nil {
		if cErr != nil {
			C.ort_shim_free_string(cErr)
		}
		return nil, ctx.Err()
	}

	if rc != 0 {
		errMsg := "unknown error"
		if cErr != nil {
			errMsg = C.GoString(cErr)
			C.ort_shim_free_string(cErr)
		}
		return nil, fmt.Errorf("ort: inference failed: %s", errMsg)
	}

	numClasses := int(outNumClasses)
	numLogits := batchSize * seqLen * numClasses
	a.Logits = a.Logits[:numLogits]

	// Decode logits into spans per window
	results := make([][]types.EntitySpan, batchSize)
	a.SpanScratch = a.SpanScratch[:0]
	for i, w := range batch.Windows {
		logitsStart := i * seqLen * numClasses
		logitsEnd := logitsStart + seqLen*numClasses
		windowLogits := a.Logits[logitsStart:logitsEnd]

		if e.decoder != nil {
			startIdx := len(a.SpanScratch)
			dst := a.SpanScratch[startIdx:startIdx]
			decoded := e.decoder.DecodeWindow(windowLogits, numClasses, w.Offsets, dst)
			results[i] = decoded
			if cap(a.SpanScratch)-startIdx >= len(decoded) {
				a.SpanScratch = a.SpanScratch[:startIdx+len(decoded)]
			}
		}
	}

	return results, nil
}
