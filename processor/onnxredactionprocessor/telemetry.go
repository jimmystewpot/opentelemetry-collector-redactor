package onnxredactionprocessor

import (
	"context"
	"time"

	"go.opentelemetry.io/collector/component"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

type telemetry struct {
	durationSeconds  metric.Float64Histogram
	batchStrings     metric.Int64Histogram
	tensorBatchSize  metric.Int64Histogram
	paddingRatio     metric.Float64Histogram
	entitiesTotal    metric.Int64Counter
	inspectedTotal   metric.Int64Counter
	skippedTotal     metric.Int64Counter
	errorsTotal      metric.Int64Counter
	scrubbedTotal    metric.Int64Counter
	droppedTotal     metric.Int64Counter
	inflight         metric.Int64UpDownCounter
	queueWait        metric.Float64Histogram
	metricCollisions metric.Int64Counter
	metricLabelPII   metric.Int64Counter
	reg              metric.Registration
}

func newTelemetry(set *component.TelemetrySettings, ortVersion, modelSHA, kernelPath string) (*telemetry, error) {
	meter := set.MeterProvider.Meter("github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor")

	durationSeconds, err := meter.Float64Histogram(
		"redaction_duration_seconds",
		metric.WithDescription("Duration of redaction pipeline stages in seconds"),
		metric.WithUnit("s"),
	)
	if err != nil {
		return nil, err
	}

	batchStrings, err := meter.Int64Histogram(
		"redaction_batch_strings",
		metric.WithDescription("Number of unique candidate strings in coalesced batches"),
		metric.WithUnit("{strings}"),
	)
	if err != nil {
		return nil, err
	}

	tensorBatchSize, err := meter.Int64Histogram(
		"redaction_tensor_batch_size",
		metric.WithDescription("Tensor sub-batch size dispatched to ONNX Runtime"),
		metric.WithUnit("{windows}"),
	)
	if err != nil {
		return nil, err
	}

	paddingRatio, err := meter.Float64Histogram(
		"redaction_padding_ratio",
		metric.WithDescription("Ratio of valid token IDs to total allocated tensor slots"),
		metric.WithUnit("1"),
	)
	if err != nil {
		return nil, err
	}

	entitiesTotal, err := meter.Int64Counter(
		"redaction_entities_total",
		metric.WithDescription("Count of detected and masked entities by label"),
		metric.WithUnit("{entities}"),
	)
	if err != nil {
		return nil, err
	}

	inspectedTotal, err := meter.Int64Counter(
		"redaction_inspected_strings_total",
		metric.WithDescription("Total strings inspected by telemetry signal"),
		metric.WithUnit("{strings}"),
	)
	if err != nil {
		return nil, err
	}

	skippedTotal, err := meter.Int64Counter(
		"redaction_skipped_total",
		metric.WithDescription("Total attributes or strings skipped by reason"),
		metric.WithUnit("{strings}"),
	)
	if err != nil {
		return nil, err
	}

	errorsTotal, err := meter.Int64Counter(
		"redaction_errors_total",
		metric.WithDescription("Total processing and inference errors by reason"),
		metric.WithUnit("{errors}"),
	)
	if err != nil {
		return nil, err
	}

	scrubbedTotal, err := meter.Int64Counter(
		"redaction_scrubbed_total",
		metric.WithDescription("Count of individual strings scrubbed in-place with [REDACTED_UNINSPECTED]"),
		metric.WithUnit("{strings}"),
	)
	if err != nil {
		return nil, err
	}

	droppedTotal, err := meter.Int64Counter(
		"redaction_dropped_total",
		metric.WithDescription("Count of unparseable records dropped by signal and reason"),
		metric.WithUnit("{records}"),
	)
	if err != nil {
		return nil, err
	}

	inflight, err := meter.Int64UpDownCounter(
		"redaction_inflight",
		metric.WithDescription("Current in-flight redaction requests"),
		metric.WithUnit("{requests}"),
	)
	if err != nil {
		return nil, err
	}

	queueWait, err := meter.Float64Histogram(
		"redaction_queue_wait_seconds",
		metric.WithDescription("Time requests spend waiting in coalescing window before inference"),
		metric.WithUnit("s"),
	)
	if err != nil {
		return nil, err
	}

	metricCollisions, err := meter.Int64Counter(
		"redaction_metric_collisions_total",
		metric.WithDescription("Count of colliding metric datapoints resolved after redaction"),
		metric.WithUnit("{collisions}"),
	)
	if err != nil {
		return nil, err
	}

	metricLabelPII, err := meter.Int64Counter(
		"redaction_metric_label_pii_total",
		metric.WithDescription("Count of metric metrics where PII label was detected and redacted"),
		metric.WithUnit("{metrics}"),
	)
	if err != nil {
		return nil, err
	}

	// Register engine info gauge
	engineInfo, err := meter.Int64ObservableGauge(
		"redaction_engine_info",
		metric.WithDescription("Static engine and model build metadata"),
		metric.WithUnit("1"),
	)
	var reg metric.Registration
	if err == nil {
		var regErr error
		reg, regErr = meter.RegisterCallback(func(_ context.Context, o metric.Observer) error {
			o.ObserveInt64(engineInfo, 1, metric.WithAttributes(
				attribute.String("ort_version", ortVersion),
				attribute.String("model_sha", modelSHA),
				attribute.String("kernel_path", kernelPath),
			))
			return nil
		}, engineInfo)
		if regErr != nil {
			return nil, regErr
		}
	}

	return &telemetry{
		durationSeconds:  durationSeconds,
		batchStrings:     batchStrings,
		tensorBatchSize:  tensorBatchSize,
		paddingRatio:     paddingRatio,
		entitiesTotal:    entitiesTotal,
		inspectedTotal:   inspectedTotal,
		skippedTotal:     skippedTotal,
		errorsTotal:      errorsTotal,
		scrubbedTotal:    scrubbedTotal,
		droppedTotal:     droppedTotal,
		inflight:         inflight,
		queueWait:        queueWait,
		metricCollisions: metricCollisions,
		metricLabelPII:   metricLabelPII,
		reg:              reg,
	}, nil
}

// Close unregisters the metric callback registration.
func (t *telemetry) Close() error {
	if t == nil || t.reg == nil {
		return nil
	}
	return t.reg.Unregister()
}

func (t *telemetry) RecordBatch(coalescedStrings int) {
	if t == nil {
		return
	}
	t.batchStrings.Record(context.Background(), int64(coalescedStrings))
}

func (t *telemetry) RecordTensorBatch(size int, paddingRatio float64) {
	if t == nil {
		return
	}
	t.tensorBatchSize.Record(context.Background(), int64(size))
	t.paddingRatio.Record(context.Background(), paddingRatio)
}

var (
	stageTokenizeOpt = metric.WithAttributes(attribute.String("stage", "tokenize"))
	stageInferOpt    = metric.WithAttributes(attribute.String("stage", "infer"))
	stageDecodeOpt   = metric.WithAttributes(attribute.String("stage", "decode"))
	stageMaskOpt     = metric.WithAttributes(attribute.String("stage", "mask"))
)

func (t *telemetry) RecordStageDuration(stage string, d time.Duration) {
	if t == nil {
		return
	}
	var opt metric.MeasurementOption
	switch stage {
	case "tokenize":
		opt = stageTokenizeOpt
	case "infer":
		opt = stageInferOpt
	case "decode":
		opt = stageDecodeOpt
	case "mask":
		opt = stageMaskOpt
	default:
		opt = metric.WithAttributes(attribute.String("stage", stage))
	}
	t.durationSeconds.Record(context.Background(), d.Seconds(), opt)
}

func (t *telemetry) RecordEntity(label types.Label) {
	if t == nil {
		return
	}
	t.entitiesTotal.Add(context.Background(), 1, metric.WithAttributes(attribute.String("label", label.String())))
}

func (t *telemetry) RecordScrubbed(signal, reason string) {
	if t == nil {
		return
	}
	if signal == "" {
		signal = "telemetry"
	}
	t.scrubbedTotal.Add(context.Background(), 1, metric.WithAttributes(
		attribute.String("signal", signal),
		attribute.String("reason", reason),
	))
}

func (t *telemetry) RecordError(reason string) {
	if t == nil {
		return
	}
	t.errorsTotal.Add(context.Background(), 1, metric.WithAttributes(attribute.String("reason", reason)))
}

func (t *telemetry) RecordInspected(signal string, count int) {
	if t == nil || count <= 0 {
		return
	}
	t.inspectedTotal.Add(context.Background(), int64(count), metric.WithAttributes(attribute.String("signal", signal)))
}

func (t *telemetry) RecordSkipped(reason string, count int) {
	if t == nil || count <= 0 {
		return
	}
	t.skippedTotal.Add(context.Background(), int64(count), metric.WithAttributes(attribute.String("reason", reason)))
}

func (t *telemetry) RecordDropped(signal, reason string, count int) {
	if t == nil || count <= 0 {
		return
	}
	t.droppedTotal.Add(context.Background(), int64(count), metric.WithAttributes(
		attribute.String("signal", signal),
		attribute.String("reason", reason),
	))
}

func (t *telemetry) RecordMetricCollision(metricType string, count int) {
	if t == nil || count <= 0 {
		return
	}
	t.metricCollisions.Add(context.Background(), int64(count), metric.WithAttributes(attribute.String("type", metricType)))
}

func (t *telemetry) RecordMetricLabelPII(metricName string, count int) {
	if t == nil || count <= 0 {
		return
	}
	t.metricLabelPII.Add(context.Background(), int64(count), metric.WithAttributes(attribute.String("metric", metricName)))
}

func (t *telemetry) RecordInflight(delta int64) {
	if t == nil {
		return
	}
	t.inflight.Add(context.Background(), delta)
}

func (t *telemetry) RecordQueueWait(d time.Duration) {
	if t == nil {
		return
	}
	t.queueWait.Record(context.Background(), d.Seconds())
}
