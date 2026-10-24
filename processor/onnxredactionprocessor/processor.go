package onnxredactionprocessor

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"go.opentelemetry.io/collector/component"
	"go.opentelemetry.io/collector/consumer"
	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/plog"
	"go.opentelemetry.io/collector/pdata/pmetric"
	"go.opentelemetry.io/collector/pdata/ptrace"
	"go.opentelemetry.io/collector/processor"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/bucket"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/coalesce"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/decode"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/dedup"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/detect"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/filter"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/metricsmerge"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/ort"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/tokenizer"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/window"
)

var (
	_ processor.Logs    = (*redactionProcessor)(nil)
	_ processor.Traces  = (*redactionProcessor)(nil)
	_ processor.Metrics = (*redactionProcessor)(nil)
)

var defaultIDToBIO = []string{
	"O",
	"B-PERSON", "B-EMAIL", "B-PHONE", "B-LOCATION", "B-CREDIT_CARD", "B-SSN", "B-IBAN", "B-MEDICAL_ID", "B-PATH_USER",
	"I-PERSON", "I-EMAIL", "I-PHONE", "I-LOCATION", "I-CREDIT_CARD", "I-SSN", "I-IBAN", "I-MEDICAL_ID", "I-PATH_USER",
}

type redactionProcessor struct {
	cfg         *Config
	telemetry   *telemetry
	coalescer   *coalesce.Coalescer
	detector    detect.Detector
	nextLogs    consumer.Logs
	nextTraces  consumer.Traces
	nextMetrics consumer.Metrics
}

func newProcessor(
	cfg *Config,
	set *component.TelemetrySettings,
	tok tokenizer.Tokenizer,
	det detect.Detector,
) (*redactionProcessor, error) {
	if cfg == nil {
		return nil, errors.New("onnxredaction: config is nil")
	}

	tel, err := newTelemetry(set, "1.31.0", cfg.Model.SHA256, "mlas")
	if err != nil {
		return nil, fmt.Errorf("onnxredaction: failed to initialize telemetry: %w", err)
	}

	var allowed [10]bool
	if len(cfg.Entities) == 0 {
		for i := 1; i <= 9; i++ {
			allowed[i] = true
		}
	} else {
		for _, e := range cfg.Entities {
			if lbl, ok := types.ParseLabel(e); ok && lbl != types.LabelNone {
				allowed[lbl] = true
			}
		}
	}

	filteredIDToBIO := make([]string, len(defaultIDToBIO))
	for i, tag := range defaultIDToBIO {
		name := strings.TrimPrefix(strings.TrimPrefix(tag, "B-"), "I-")
		lbl, ok := types.ParseLabel(name)
		if ok && allowed[lbl] {
			filteredIDToBIO[i] = tag
		} else {
			filteredIDToBIO[i] = "O"
		}
	}

	if tok == nil {
		if cfg.Model.TokenizerPath == "" {
			return nil, errors.New("onnxredaction: model.tokenizer_path is required")
		}
		tok, err = tokenizer.NewWordPieceTokenizer(cfg.Model.TokenizerPath, cfg.Model.TokenizerSHA256)
		if err != nil {
			return nil, fmt.Errorf("onnxredaction: failed to load tokenizer: %w", err)
		}
	}

	if det == nil {
		if cfg.Model.Path == "" {
			return nil, errors.New("onnxredaction: model.path is required")
		}
		dec := decode.NewDecoder(filteredIDToBIO, nil, float32(cfg.Inference.ConfidenceThreshold))
		det, err = ort.NewEngine(
			cfg.Model.Path,
			cfg.Model.SHA256,
			cfg.Model.Session.IntraOpThreads,
			cfg.Model.Session.ConcurrentRuns,
			cfg.Model.Session.AllowSpinning,
			dec,
		)
		if err != nil {
			return nil, fmt.Errorf("onnxredaction: failed to initialize ORT engine: %w", err)
		}
	}

	filt := filter.New(cfg.Filtering.MinStringLength, cfg.Filtering.ExcludeAttributes)
	chunker := window.NewChunker(cfg.Inference.WindowTokens, cfg.Inference.WindowOverlapTokens, cfg.Inference.MaxPayloadBytes)
	bucketer := bucket.New(cfg.Inference.LengthBuckets, cfg.Inference.MaxTensorBatchSize)
	arenaPool := arena.NewPool(cfg.Inference.MaxTensorBatchSize, cfg.Inference.WindowTokens, len(defaultIDToBIO))

	c := coalesce.New(
		cfg.Inference.CoalesceWindow,
		cfg.Inference.CoalesceMaxStrings,
		filt,
		tok,
		chunker,
		bucketer,
		det,
		arenaPool,
		cfg.Inference.MaxPayloadBytes,
		cfg.OnError,
		cfg.DryRun,
		coalesce.WithTimeout(cfg.Inference.Timeout),
		coalesce.WithTelemetry(tel),
		coalesce.WithConcurrentRuns(cfg.Model.Session.ConcurrentRuns),
		coalesce.WithAllowedLabels(allowed),
		coalesce.WithLogger(set.Logger),
	)

	return &redactionProcessor{
		cfg:       cfg,
		telemetry: tel,
		coalescer: c,
		detector:  det,
	}, nil
}

func (p *redactionProcessor) Start(ctx context.Context, host component.Host) error {
	if p.cfg.Inference.WarmupOnStart && p.detector != nil {
		if err := p.detector.Warmup(ctx); err != nil {
			return fmt.Errorf("onnxredaction: detector warmup failed: %w", err)
		}
	}
	return nil
}

func (p *redactionProcessor) Shutdown(ctx context.Context) error {
	var errs []error
	if p.coalescer != nil {
		if err := p.coalescer.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.detector != nil {
		if err := p.detector.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if p.telemetry != nil {
		if err := p.telemetry.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (p *redactionProcessor) Capabilities() consumer.Capabilities {
	return consumer.Capabilities{MutatesData: true}
}

func (p *redactionProcessor) processLogs(ctx context.Context, ld plog.Logs) error {
	if ld.LogRecordCount() == 0 {
		return nil
	}

	items := make([]coalesce.Item, 0, ld.LogRecordCount()*2)
	rLogs := ld.ResourceLogs()
	for i := 0; i < rLogs.Len(); i++ {
		rl := rLogs.At(i)
		p.collectMapCandidates("logs", rl.Resource().Attributes(), &items)
		sLogs := rl.ScopeLogs()
		for j := 0; j < sLogs.Len(); j++ {
			sl := sLogs.At(j)
			p.collectMapCandidates("logs", sl.Scope().Attributes(), &items)
			records := sl.LogRecords()
			for k := 0; k < records.Len(); k++ {
				rec := records.At(k)
				p.collectMapCandidates("logs", rec.Attributes(), &items)
				body := rec.Body()
				if body.Type() == pcommon.ValueTypeBytes {
					p.telemetry.RecordSkipped("bytes_body", 1)
				} else {
					p.collectValueCandidates("logs", "body", body, &items, 0)
				}
			}
		}
	}

	p.telemetry.RecordInspected("logs", len(items))
	return p.coalescer.Submit(ctx, items)
}

func (p *redactionProcessor) ConsumeLogs(ctx context.Context, ld plog.Logs) error {
	if err := p.processLogs(ctx, ld); err != nil {
		return err
	}
	if p.nextLogs != nil {
		return p.nextLogs.ConsumeLogs(ctx, ld)
	}
	return nil
}

func (p *redactionProcessor) processTraces(ctx context.Context, td ptrace.Traces) error {
	if td.SpanCount() == 0 {
		return nil
	}

	items := make([]coalesce.Item, 0, td.SpanCount()*3)
	rSpans := td.ResourceSpans()
	for i := 0; i < rSpans.Len(); i++ {
		rs := rSpans.At(i)
		p.collectMapCandidates("traces", rs.Resource().Attributes(), &items)
		sSpans := rs.ScopeSpans()
		for j := 0; j < sSpans.Len(); j++ {
			ss := sSpans.At(j)
			p.collectMapCandidates("traces", ss.Scope().Attributes(), &items)
			spans := ss.Spans()
			for k := 0; k < spans.Len(); k++ {
				span := spans.At(k)
				items = append(items, coalesce.Item{
					Signal: "traces",
					Key:    "span.name",
					Text:   span.Name(),
					Target: dedup.MutationTarget{Kind: dedup.TargetSpanName, Span: span, Signal: "traces"},
				})
				if msg := span.Status().Message(); msg != "" {
					items = append(items, coalesce.Item{
						Signal: "traces",
						Key:    "status.message",
						Text:   msg,
						Target: dedup.MutationTarget{Kind: dedup.TargetSpanStatusMessage, Span: span, Signal: "traces"},
					})
				}
				p.collectMapCandidates("traces", span.Attributes(), &items)
				events := span.Events()
				for e := 0; e < events.Len(); e++ {
					ev := events.At(e)
					items = append(items, coalesce.Item{
						Signal: "traces",
						Key:    "event.name",
						Text:   ev.Name(),
						Target: dedup.MutationTarget{Kind: dedup.TargetSpanEventName, SpanEvent: ev, Signal: "traces"},
					})
					p.collectMapCandidates("traces", ev.Attributes(), &items)
				}
				links := span.Links()
				for l := 0; l < links.Len(); l++ {
					link := links.At(l)
					p.collectMapCandidates("traces", link.Attributes(), &items)
				}
			}
		}
	}

	p.telemetry.RecordInspected("traces", len(items))
	return p.coalescer.Submit(ctx, items)
}

func (p *redactionProcessor) ConsumeTraces(ctx context.Context, td ptrace.Traces) error {
	if err := p.processTraces(ctx, td); err != nil {
		return err
	}
	if p.nextTraces != nil {
		return p.nextTraces.ConsumeTraces(ctx, td)
	}
	return nil
}

func (p *redactionProcessor) processMetrics(ctx context.Context, md pmetric.Metrics) error {
	if md.DataPointCount() == 0 {
		return nil
	}

	items := make([]coalesce.Item, 0, md.DataPointCount()*2)
	rMetrics := md.ResourceMetrics()
	for i := 0; i < rMetrics.Len(); i++ {
		rm := rMetrics.At(i)
		p.collectMapCandidates("metrics", rm.Resource().Attributes(), &items)
		sMetrics := rm.ScopeMetrics()
		for j := 0; j < sMetrics.Len(); j++ {
			sm := sMetrics.At(j)
			p.collectMapCandidates("metrics", sm.Scope().Attributes(), &items)
			metrics := sm.Metrics()
			for k := 0; k < metrics.Len(); k++ {
				m := metrics.At(k)
				items = append(items,
					coalesce.Item{
						Signal: "metrics",
						Key:    "metric.name",
						Text:   m.Name(),
						Target: dedup.MutationTarget{Kind: dedup.TargetMetricName, Metric: m, MetricName: m.Name(), Signal: "metrics"},
					},
					coalesce.Item{
						Signal: "metrics",
						Key:    "metric.description",
						Text:   m.Description(),
						Target: dedup.MutationTarget{Kind: dedup.TargetMetricDescription, Metric: m, MetricName: m.Name(), Signal: "metrics"},
					},
				)
				p.collectDataPointCandidates(m, &items)
			}
		}
	}

	p.telemetry.RecordInspected("metrics", len(items))

	// Snapshot pre-redaction datapoint attribute hashes
	metricHashes := make([]uint64, 0, md.MetricCount())
	var scratch []string
	for i := 0; i < rMetrics.Len(); i++ {
		rm := rMetrics.At(i)
		sMetrics := rm.ScopeMetrics()
		for j := 0; j < sMetrics.Len(); j++ {
			sm := sMetrics.At(j)
			metrics := sm.Metrics()
			for k := 0; k < metrics.Len(); k++ {
				var h uint64
				h, scratch = metricDataPointsHash(metrics.At(k), scratch)
				metricHashes = append(metricHashes, h)
			}
		}
	}

	if err := p.coalescer.Submit(ctx, items); err != nil {
		return err
	}

	// After redaction, resolve collisions only across metrics whose datapoint attributes changed
	var idx int
	for i := 0; i < rMetrics.Len(); i++ {
		rm := rMetrics.At(i)
		sMetrics := rm.ScopeMetrics()
		for j := 0; j < sMetrics.Len(); j++ {
			sm := sMetrics.At(j)
			metrics := sm.Metrics()
			for k := 0; k < metrics.Len(); k++ {
				m := metrics.At(k)
				var postHash uint64
				postHash, scratch = metricDataPointsHash(m, scratch)
				if idx < len(metricHashes) && postHash != metricHashes[idx] {
					stats := metricsmerge.ResolveCollisions(m)
					for mType, cnt := range stats.CollisionsByType {
						p.telemetry.RecordMetricCollision(mType.String(), cnt)
					}
				}
				idx++
			}
		}
	}
	return nil
}

func (p *redactionProcessor) ConsumeMetrics(ctx context.Context, md pmetric.Metrics) error {
	if err := p.processMetrics(ctx, md); err != nil {
		return err
	}
	if p.nextMetrics != nil {
		return p.nextMetrics.ConsumeMetrics(ctx, md)
	}
	return nil
}

func metricDataPointsHash(m pmetric.Metric, scratch []string) (uint64, []string) {
	var h uint64 = 14695981039346656037
	hashDP := func(attrs pcommon.Map, idx int) {
		var dpHash uint64
		dpHash, scratch = metricsmerge.ComputeCanonicalAttrHash(attrs, scratch)
		h = (h ^ (dpHash + uint64(idx))) * 1099511628211 //nolint:gosec // Loop index is non-negative
	}
	switch m.Type() {
	case pmetric.MetricTypeGauge:
		dps := m.Gauge().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			hashDP(dps.At(i).Attributes(), i)
		}
	case pmetric.MetricTypeSum:
		dps := m.Sum().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			hashDP(dps.At(i).Attributes(), i)
		}
	case pmetric.MetricTypeHistogram:
		dps := m.Histogram().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			hashDP(dps.At(i).Attributes(), i)
		}
	case pmetric.MetricTypeExponentialHistogram:
		dps := m.ExponentialHistogram().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			hashDP(dps.At(i).Attributes(), i)
		}
	case pmetric.MetricTypeSummary:
		dps := m.Summary().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			hashDP(dps.At(i).Attributes(), i)
		}
	}
	return h, scratch
}

const maxNestingDepth = 32

func (p *redactionProcessor) scrubValue(signal string, val pcommon.Value) {
	switch val.Type() {
	case pcommon.ValueTypeStr:
		val.SetStr(types.RedactedUninspected)
		p.telemetry.RecordScrubbed(signal, "max_depth")
	case pcommon.ValueTypeMap:
		val.Map().Range(func(_ string, v pcommon.Value) bool {
			p.scrubValue(signal, v)
			return true
		})
	case pcommon.ValueTypeSlice:
		slice := val.Slice()
		for i := 0; i < slice.Len(); i++ {
			p.scrubValue(signal, slice.At(i))
		}
	}
}

func (p *redactionProcessor) collectMapCandidates(signal string, m pcommon.Map, items *[]coalesce.Item) {
	p.collectMapCandidatesWithMetric(signal, m, items, "")
}

func (p *redactionProcessor) collectMapCandidatesWithMetric(signal string, m pcommon.Map, items *[]coalesce.Item, metricName string) {
	m.Range(func(k string, v pcommon.Value) bool {
		p.collectValueCandidatesWithMetric(signal, k, v, items, 0, metricName)
		return true
	})
}

func (p *redactionProcessor) collectValueCandidates(signal, key string, val pcommon.Value, items *[]coalesce.Item, depth int) {
	p.collectValueCandidatesWithMetric(signal, key, val, items, depth, "")
}

func (p *redactionProcessor) collectValueCandidatesWithMetric(signal, key string, val pcommon.Value, items *[]coalesce.Item, depth int, metricName string) {
	if depth > maxNestingDepth {
		p.telemetry.RecordSkipped("max_depth", 1)
		if p.cfg.OnError == types.ErrorModeDrop {
			p.scrubValue(signal, val)
		}
		return
	}
	switch val.Type() {
	case pcommon.ValueTypeStr:
		*items = append(*items, coalesce.Item{
			Signal: signal,
			Key:    key,
			Text:   val.Str(),
			Target: dedup.MutationTarget{Kind: dedup.TargetPCommonValue, Val: val, MetricName: metricName, Signal: signal},
		})
	case pcommon.ValueTypeMap:
		val.Map().Range(func(k string, v pcommon.Value) bool {
			p.collectValueCandidatesWithMetric(signal, k, v, items, depth+1, metricName)
			return true
		})
	case pcommon.ValueTypeSlice:
		slice := val.Slice()
		for i := 0; i < slice.Len(); i++ {
			p.collectValueCandidatesWithMetric(signal, key, slice.At(i), items, depth+1, metricName)
		}
	}
}

func (p *redactionProcessor) collectDataPointCandidates(m pmetric.Metric, items *[]coalesce.Item) {
	name := m.Name()
	switch m.Type() {
	case pmetric.MetricTypeGauge:
		dps := m.Gauge().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			p.collectMapCandidatesWithMetric("metrics", dp.Attributes(), items, name)
			exemplars := dp.Exemplars()
			for e := 0; e < exemplars.Len(); e++ {
				p.collectMapCandidatesWithMetric("metrics", exemplars.At(e).FilteredAttributes(), items, name)
			}
		}
	case pmetric.MetricTypeSum:
		dps := m.Sum().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			p.collectMapCandidatesWithMetric("metrics", dp.Attributes(), items, name)
			exemplars := dp.Exemplars()
			for e := 0; e < exemplars.Len(); e++ {
				p.collectMapCandidatesWithMetric("metrics", exemplars.At(e).FilteredAttributes(), items, name)
			}
		}
	case pmetric.MetricTypeHistogram:
		dps := m.Histogram().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			p.collectMapCandidatesWithMetric("metrics", dp.Attributes(), items, name)
			exemplars := dp.Exemplars()
			for e := 0; e < exemplars.Len(); e++ {
				p.collectMapCandidatesWithMetric("metrics", exemplars.At(e).FilteredAttributes(), items, name)
			}
		}
	case pmetric.MetricTypeExponentialHistogram:
		dps := m.ExponentialHistogram().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			p.collectMapCandidatesWithMetric("metrics", dp.Attributes(), items, name)
			exemplars := dp.Exemplars()
			for e := 0; e < exemplars.Len(); e++ {
				p.collectMapCandidatesWithMetric("metrics", exemplars.At(e).FilteredAttributes(), items, name)
			}
		}
	case pmetric.MetricTypeSummary:
		dps := m.Summary().DataPoints()
		for i := 0; i < dps.Len(); i++ {
			p.collectMapCandidatesWithMetric("metrics", dps.At(i).Attributes(), items, name)
		}
	}
}
