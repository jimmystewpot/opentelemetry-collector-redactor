package coalesce

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
	"unicode/utf8"

	"go.uber.org/zap"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/bucket"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/decode"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/dedup"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/detect"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/filter"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/mask"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/tokenizer"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/window"
)

// Item represents a single candidate string submitted for NER redaction.
type Item struct {
	Signal string
	Key    string
	Text   string
	Target dedup.MutationTarget
}

// TelemetryRecorder records internal pipeline events and metrics.
type TelemetryRecorder interface {
	RecordBatch(coalescedStrings int)
	RecordTensorBatch(size int, paddingRatio float64)
	RecordStageDuration(stage string, d time.Duration)
	RecordEntity(label types.Label)
	RecordScrubbed(signal string, reason string)
	RecordError(reason string)
	RecordSkipped(reason string, count int)
	RecordDropped(signal string, reason string, count int)
	RecordInflight(delta int64)
	RecordQueueWait(d time.Duration)
	RecordMetricCollision(metricType string, count int)
	RecordMetricLabelPII(metricName string, count int)
}

type noopTelemetry struct{}

func (noopTelemetry) RecordBatch(int)                           {}
func (noopTelemetry) RecordTensorBatch(int, float64)            {}
func (noopTelemetry) RecordStageDuration(string, time.Duration) {}
func (noopTelemetry) RecordEntity(types.Label)                  {}
func (noopTelemetry) RecordScrubbed(string, string)             {}
func (noopTelemetry) RecordError(string)                        {}
func (noopTelemetry) RecordSkipped(string, int)                 {}
func (noopTelemetry) RecordDropped(string, string, int)         {}
func (noopTelemetry) RecordInflight(int64)                      {}
func (noopTelemetry) RecordQueueWait(time.Duration)             {}
func (noopTelemetry) RecordMetricCollision(string, int)         {}
func (noopTelemetry) RecordMetricLabelPII(string, int)          {}

// Option configures optional parameters for Coalescer.
type Option func(*Coalescer)

// WithTimeout sets the per-sub-batch detector timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Coalescer) {
		c.timeout = d
	}
}

// WithTelemetry sets the metrics recorder for the coalescer.
func WithTelemetry(t TelemetryRecorder) Option {
	return func(c *Coalescer) {
		if t != nil {
			c.telemetry = t
		}
	}
}

// WithConcurrentRuns limits the maximum number of concurrent batch inference executions.
func WithConcurrentRuns(n int) Option {
	return func(c *Coalescer) {
		if n > 0 {
			c.sem = make(chan struct{}, n)
		}
	}
}

// WithAllowedLabels filters detected entity spans to only labels enabled in the mask.
func WithAllowedLabels(allowed [10]bool) Option {
	return func(c *Coalescer) {
		c.allowedLabels = allowed
		c.hasAllowedLabels = true
	}
}

// WithLogger sets the logger for the coalescer.
func WithLogger(logger *zap.Logger) Option {
	return func(c *Coalescer) {
		c.logger = logger
	}
}

type callerRequest struct {
	ctx         context.Context
	items       []Item
	done        chan struct{}
	err         error
	removed     bool
	submittedAt time.Time
}

// Coalescer aggregates concurrent string redaction submissions over a sliding
// time window up to a maximum batch size, executing inference as a unified batch.
type Coalescer struct {
	window           time.Duration
	maxStrings       int
	filter           *filter.AttributeFilter
	tokenizer        tokenizer.Tokenizer
	chunker          *window.Chunker
	bucketer         *bucket.Bucketer
	detector         detect.Detector
	arenaPool        *arena.Pool
	maxPayloadBytes  int
	errorMode        types.ErrorMode
	dryRun           bool
	timeout          time.Duration
	telemetry        TelemetryRecorder
	allowedLabels    [10]bool
	hasAllowedLabels bool
	logger           *zap.Logger

	mu           sync.Mutex
	pending      []*callerRequest
	pendingCount int
	timer        *time.Timer
	timerActive  bool
	closed       bool
	activeWG     sync.WaitGroup
	sem          chan struct{}
}

// New creates a new Coalescer with the provided configuration and pipeline stages.
func New(
	coalesceWindow time.Duration,
	maxStrings int,
	filt *filter.AttributeFilter,
	tok tokenizer.Tokenizer,
	chunker *window.Chunker,
	bucketer *bucket.Bucketer,
	detector detect.Detector,
	arenaPool *arena.Pool,
	maxPayloadBytes int,
	errorMode types.ErrorMode,
	dryRun bool,
	opts ...Option,
) *Coalescer {
	if coalesceWindow <= 0 {
		coalesceWindow = 2 * time.Millisecond
	}
	if maxStrings <= 0 {
		maxStrings = 64
	}
	if maxPayloadBytes <= 0 {
		maxPayloadBytes = 16384
	}

	c := &Coalescer{
		window:          coalesceWindow,
		maxStrings:      maxStrings,
		filter:          filt,
		tokenizer:       tok,
		chunker:         chunker,
		bucketer:        bucketer,
		detector:        detector,
		arenaPool:       arenaPool,
		maxPayloadBytes: maxPayloadBytes,
		errorMode:       errorMode,
		dryRun:          dryRun,
		timeout:         25 * time.Millisecond,
		telemetry:       noopTelemetry{},
		sem:             make(chan struct{}, 2),
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// Submit queues items for batched redaction and blocks synchronously until the batch
// completes or the caller context is canceled.
func (c *Coalescer) Submit(ctx context.Context, items []Item) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if len(items) == 0 {
		return nil
	}

	c.telemetry.RecordInflight(1)
	defer c.telemetry.RecordInflight(-1)

	req := &callerRequest{
		ctx:         ctx,
		items:       items,
		done:        make(chan struct{}),
		submittedAt: time.Now(),
	}

	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return errors.New("coalescer is closed")
	}

	c.pending = append(c.pending, req)
	c.pendingCount += len(items)

	switch {
	case c.pendingCount >= c.maxStrings:
		if c.timerActive {
			c.timer.Stop()
			c.timerActive = false
		}
		batchToRun := c.pending
		c.pending = nil
		c.pendingCount = 0
		c.activeWG.Add(1)
		c.mu.Unlock()

		go c.executeBatch(batchToRun) //nolint:gosec // executeBatch is an asynchronous coalesced runner across multiple requests
	case !c.timerActive:
		if c.timer == nil {
			c.timer = time.AfterFunc(c.window, c.onTimer)
		} else {
			c.timer.Reset(c.window)
		}
		c.timerActive = true
		c.mu.Unlock()
	default:
		c.mu.Unlock()
	}

	select {
	case <-ctx.Done():
		c.mu.Lock()
		idx := -1
		for i, p := range c.pending {
			if p == req {
				idx = i
				break
			}
		}
		if idx != -1 {
			// Pre-batch cancellation: safely unregister
			c.pending = append(c.pending[:idx], c.pending[idx+1:]...)
			c.pendingCount -= len(req.items)
			req.removed = true
			if len(c.pending) == 0 && c.timerActive {
				c.timer.Stop()
				c.timerActive = false
			}
			c.mu.Unlock()
			return ctx.Err()
		}
		c.mu.Unlock()

		// Batch has already started inference: must block until CGo/inference finishes
		// to guarantee caller pdata is never accessed concurrently.
		<-req.done
		return ctx.Err()

	case <-req.done:
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return req.err
	}
}

func (c *Coalescer) onTimer() {
	c.mu.Lock()
	if c.closed || len(c.pending) == 0 {
		c.timerActive = false
		c.mu.Unlock()
		return
	}

	batchToRun := c.pending
	c.pending = nil
	c.pendingCount = 0
	c.timerActive = false
	c.activeWG.Add(1)
	c.mu.Unlock()

	go c.executeBatch(batchToRun)
}

func (c *Coalescer) executeBatch(batch []*callerRequest) {
	defer c.activeWG.Done()
	defer func() {
		if r := recover(); r != nil {
			if c.logger != nil {
				c.logger.Error("redaction batch panic recovered", zap.Any("panic", r), zap.Stack("stacktrace"))
			}
			c.telemetry.RecordError("panic")
			panicErr := fmt.Errorf("redaction batch panic: %v", r)
			for _, req := range batch {
				if req.err == nil {
					req.err = panicErr
					if c.errorMode == types.ErrorModeDrop && len(req.items) > 0 {
						c.telemetry.RecordDropped(req.items[0].Signal, "panic", len(req.items))
					}
				}
			}
		}
		for _, req := range batch {
			close(req.done)
		}
	}()

	activeRequests := make([]*callerRequest, 0, len(batch))
	for _, req := range batch {
		if !req.removed {
			activeRequests = append(activeRequests, req)
		}
	}
	if len(activeRequests) == 0 {
		return
	}

	for _, req := range activeRequests {
		c.telemetry.RecordQueueWait(time.Since(req.submittedAt))
	}

	anyCallerAlive := func() bool {
		for _, req := range activeRequests {
			if req.ctx.Err() == nil {
				return true
			}
		}
		return false
	}
	if !anyCallerAlive() {
		return
	}

	// Concurrency bounding semaphore
	if c.sem != nil {
		select {
		case c.sem <- struct{}{}:
			defer func() { <-c.sem }()
		default:
			c.telemetry.RecordError("semaphore_full")
			if c.errorMode == types.ErrorModeDrop {
				err := errors.New("redaction worker semaphore full (inference overloaded)")
				for _, req := range activeRequests {
					req.err = err
					if len(req.items) > 0 {
						c.telemetry.RecordDropped(req.items[0].Signal, "semaphore_full", len(req.items))
					}
				}
				return
			}
			return
		}
	}

	a := c.arenaPool.Acquire()
	defer c.arenaPool.Release(a)
	a.Dedup.Reset()

	// 1. Filter & Deduplicate
	for _, req := range activeRequests {
		for itemIdx := range req.items {
			item := &req.items[itemIdx]
			if c.filter != nil && !c.filter.ShouldInspect(item.Key, item.Text) {
				c.telemetry.RecordSkipped("filter", 1)
				continue
			}
			if !utf8.ValidString(item.Text) {
				c.telemetry.RecordError("invalid_utf8")
				if c.errorMode == types.ErrorModeDrop && !c.dryRun {
					item.Target.Set(types.RedactedUninspected)
					c.telemetry.RecordScrubbed(item.Signal, "invalid_utf8")
				}
				continue
			}
			trg := item.Target
			if trg.Signal == "" && item.Signal != "" {
				trg.Signal = item.Signal
			}
			a.Dedup.Add(item.Text, trg)
		}
	}

	uniqueStrings := a.Dedup.UniqueStrings()
	c.telemetry.RecordBatch(len(uniqueStrings))
	if len(uniqueStrings) == 0 {
		return
	}

	// 2. Tokenize & Window
	t0 := time.Now()
	a.Windows = a.Windows[:0]
	a.TokenIDPool = a.TokenIDPool[:0]
	a.OffsetPool = a.OffsetPool[:0]

	type parentMeta struct {
		inspectedPrefix string
		hasTail         bool
		failed          bool
	}
	meta := make([]parentMeta, len(uniqueStrings))

	for idx, text := range uniqueStrings {
		prefix, hasTail := c.chunker.SliceInput(text)
		meta[idx] = parentMeta{inspectedPrefix: prefix, hasTail: hasTail}

		if prefix == "" {
			continue
		}

		startID := len(a.TokenIDPool)
		startOffset := len(a.OffsetPool)
		ids, offsets, err := c.tokenizer.Encode(prefix, a.TokenIDPool, a.OffsetPool)
		if err != nil {
			meta[idx].failed = true
			c.telemetry.RecordError("tokenizer_error")
			if c.errorMode == types.ErrorModeDrop && !c.dryRun {
				for _, trg := range a.Dedup.TargetsFor(idx) {
					trg.Set(types.RedactedUninspected)
					c.telemetry.RecordScrubbed(trg.Signal, "tokenizer_error")
				}
			}
			continue
		}

		a.TokenIDPool = ids
		a.OffsetPool = offsets
		tokenSlice := ids[startID:]
		offsetSlice := offsets[startOffset:]

		a.Windows = c.chunker.WindowTokens(idx, tokenSlice, offsetSlice, a.Windows)
	}
	c.telemetry.RecordStageDuration("tokenize", time.Since(t0))

	// 3. Length Bucketing & Inference
	parentSpans := make([][]types.EntitySpan, len(uniqueStrings))
	if len(a.Windows) > 0 {
		tInfer := time.Now()
		batches := c.bucketer.Group(a.Windows)

		for _, b := range batches {
			if !anyCallerAlive() {
				return
			}
			actualTokens := 0
			for _, w := range b.Windows {
				actualTokens += len(w.TokenIDs)
			}
			paddedTokens := len(b.Windows) * b.MaxSeqLen
			paddingRatio := 1.0
			if paddedTokens > 0 {
				paddingRatio = float64(actualTokens) / float64(paddedTokens)
			}
			c.telemetry.RecordTensorBatch(len(b.Windows), paddingRatio)

			ctx, cancel := context.WithCancel(context.Background())
			if c.timeout > 0 {
				ctx, cancel = context.WithTimeout(context.Background(), c.timeout)
			}

			stopMonitor := make(chan struct{})
			go func() {
				defer cancel()
				for _, req := range activeRequests {
					select {
					case <-stopMonitor:
						return
					case <-ctx.Done():
						return
					case <-req.ctx.Done():
					}
				}
			}()

			spansSlice, err := c.detector.DetectBatch(ctx, b, a)
			close(stopMonitor)
			cancel()

			if err != nil {
				c.telemetry.RecordError("detector_error")
				if c.errorMode == types.ErrorModeDrop {
					for _, req := range activeRequests {
						req.err = err
						if len(req.items) > 0 {
							c.telemetry.RecordDropped(req.items[0].Signal, "detector_error", len(req.items))
						}
					}
					return
				}
				return
			}

			for wIdx, w := range b.Windows {
				pIdx := w.ParentUniqueIdx
				if wIdx < len(spansSlice) && len(spansSlice[wIdx]) > 0 {
					if c.hasAllowedLabels {
						for _, s := range spansSlice[wIdx] {
							if int(s.Label) < len(c.allowedLabels) && c.allowedLabels[s.Label] {
								parentSpans[pIdx] = append(parentSpans[pIdx], s)
							}
						}
					} else {
						parentSpans[pIdx] = append(parentSpans[pIdx], spansSlice[wIdx]...)
					}
				}
			}
		}
		c.telemetry.RecordStageDuration("infer", time.Since(tInfer))
	}

	// 4. Decode & Merge Spans
	tDecode := time.Now()
	for idx := range uniqueStrings {
		if len(parentSpans[idx]) > 0 {
			parentSpans[idx] = decode.MergeSpans(parentSpans[idx])
			for _, s := range parentSpans[idx] {
				c.telemetry.RecordEntity(s.Label)
			}
		}
	}
	c.telemetry.RecordStageDuration("decode", time.Since(tDecode))

	// 5. Mask & Mutate Targets
	tMask := time.Now()
	for idx, text := range uniqueStrings {
		if meta[idx].failed {
			continue
		}
		spans := parentSpans[idx]

		if c.dryRun {
			continue
		}

		maskedText, err := mask.Mask(text, spans, c.maxPayloadBytes, c.errorMode, a.ScratchBytes)
		if err != nil {
			if c.errorMode == types.ErrorModeDrop {
				maskedText = types.RedactedUninspected
			} else {
				maskedText = text
			}
		}

		for _, trg := range a.Dedup.TargetsFor(idx) {
			if err != nil && c.errorMode == types.ErrorModeDrop {
				c.telemetry.RecordScrubbed(trg.Signal, "mask_error")
			} else if meta[idx].hasTail && c.errorMode == types.ErrorModeDrop {
				c.telemetry.RecordScrubbed(trg.Signal, "oversize_tail")
			}
			trg.Set(maskedText)
			if trg.MetricName != "" && maskedText != text {
				c.telemetry.RecordMetricLabelPII(trg.MetricName, 1)
			}
		}
	}
	c.telemetry.RecordStageDuration("mask", time.Since(tMask))
}

// Close gracefully stops the coalescer, flushing or cancelling any pending requests.
func (c *Coalescer) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	if c.timerActive {
		c.timer.Stop()
		c.timerActive = false
	}
	for _, req := range c.pending {
		req.err = errors.New("coalescer closed")
		close(req.done)
	}
	c.pending = nil
	c.pendingCount = 0
	c.mu.Unlock()

	c.activeWG.Wait()
	return nil
}
