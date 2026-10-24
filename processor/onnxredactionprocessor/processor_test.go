package onnxredactionprocessor

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/consumer/consumererror"
	"go.opentelemetry.io/collector/consumer/consumertest"
	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/plog"
	"go.opentelemetry.io/collector/pdata/pmetric"
	"go.opentelemetry.io/collector/pdata/ptrace"
	"go.opentelemetry.io/collector/processor/processortest"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/detect"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/tokenizer"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

type mockTokenizer struct{}

func (m *mockTokenizer) Encode(text string, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset, error) {
	dstIDs = append(dstIDs, 101)
	dstOffsets = append(dstOffsets, types.Offset{Start: 0, End: len(text)})
	return dstIDs, dstOffsets, nil
}

func (m *mockTokenizer) Close() error { return nil }

type ruleDetector struct {
	rules       map[string]types.Label
	errToReturn error
}

func (r *ruleDetector) DetectBatch(_ context.Context, batch types.BucketedBatch, a *arena.BatchArena) ([][]types.EntitySpan, error) {
	if r.errToReturn != nil {
		return nil, r.errToReturn
	}

	res := make([][]types.EntitySpan, len(batch.Windows))
	for i, w := range batch.Windows {
		parentText := a.Dedup.UniqueStrings()[w.ParentUniqueIdx]
		for substr, lbl := range r.rules {
			start := strings.Index(parentText, substr)
			if start != -1 {
				res[i] = append(res[i], types.EntitySpan{
					Start:      start,
					End:        start + len(substr),
					Label:      lbl,
					Confidence: 0.99,
				})
			}
		}
	}
	return res, nil
}

func (r *ruleDetector) Warmup(_ context.Context) error { return nil }
func (r *ruleDetector) Close() error                   { return nil }

func newTestProcessor(cfg *Config, det detect.Detector) (*redactionProcessor, error) {
	set := processortest.NewNopSettings(Type)
	tok := &mockTokenizer{}
	return newProcessor(cfg, &set.TelemetrySettings, tok, det)
}

func TestProcessor_LogsRedaction(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	cfg.Filtering.MinStringLength = 1
	cfg.Filtering.ExcludeAttributes = nil

	rules := map[string]types.Label{
		"alice@example.com": types.LabelEmail,
		"John Doe":          types.LabelPerson,
		"marie.curie":       types.LabelPathUser,
	}
	proc, err := newTestProcessor(cfg, &ruleDetector{rules: rules})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	sink := new(consumertest.LogsSink)
	proc.nextLogs = sink

	ld := plog.NewLogs()
	rl := ld.ResourceLogs().AppendEmpty()
	rl.Resource().Attributes().PutStr("host.owner", "John Doe")

	sl := rl.ScopeLogs().AppendEmpty()
	rec := sl.LogRecords().AppendEmpty()
	rec.Attributes().PutStr("path", "/Users/marie.curie/.ssh/id_rsa")

	// Map body
	rec.Body().SetEmptyMap()
	rec.Body().Map().PutStr("msg", "alice@example.com logged in")

	err = proc.ConsumeLogs(context.Background(), ld)
	require.NoError(t, err)

	require.Len(t, sink.AllLogs(), 1)
	outLd := sink.AllLogs()[0]
	outRl := outLd.ResourceLogs().At(0)
	ownerVal, ok := outRl.Resource().Attributes().Get("host.owner")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_PERSON]", ownerVal.Str())

	outRec := outRl.ScopeLogs().At(0).LogRecords().At(0)
	pathVal, ok := outRec.Attributes().Get("path")
	require.True(t, ok)
	assert.Equal(t, "/Users/[REDACTED_PATH_USER]/.ssh/id_rsa", pathVal.Str())

	msgVal, ok := outRec.Body().Map().Get("msg")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_EMAIL] logged in", msgVal.Str())
}

func TestProcessor_TracesRedaction(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	cfg.Filtering.MinStringLength = 1
	cfg.Filtering.ExcludeAttributes = nil

	rules := map[string]types.Label{
		"alice@example.com": types.LabelEmail,
		"John Doe":          types.LabelPerson,
	}
	proc, err := newTestProcessor(cfg, &ruleDetector{rules: rules})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	sink := new(consumertest.TracesSink)
	proc.nextTraces = sink

	td := ptrace.NewTraces()
	rs := td.ResourceSpans().AppendEmpty()
	span := rs.ScopeSpans().AppendEmpty().Spans().AppendEmpty()
	span.SetName("GET /users/alice@example.com")
	span.Attributes().PutStr("user", "John Doe")

	ev := span.Events().AppendEmpty()
	ev.SetName("event_alice@example.com")
	ev.Attributes().PutStr("ev_user", "John Doe")

	link := span.Links().AppendEmpty()
	link.Attributes().PutStr("link_email", "alice@example.com")

	err = proc.ConsumeTraces(context.Background(), td)
	require.NoError(t, err)

	require.Len(t, sink.AllTraces(), 1)
	outSpan := sink.AllTraces()[0].ResourceSpans().At(0).ScopeSpans().At(0).Spans().At(0)
	assert.Equal(t, "GET /users/[REDACTED_EMAIL]", outSpan.Name())

	userVal, ok := outSpan.Attributes().Get("user")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_PERSON]", userVal.Str())

	outEv := outSpan.Events().At(0)
	assert.Equal(t, "event_[REDACTED_EMAIL]", outEv.Name())
	evVal, ok := outEv.Attributes().Get("ev_user")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_PERSON]", evVal.Str())

	outLink := outSpan.Links().At(0)
	linkVal, ok := outLink.Attributes().Get("link_email")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_EMAIL]", linkVal.Str())
}

func TestProcessor_MetricsRedactionAndCollision(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	cfg.Filtering.MinStringLength = 1
	cfg.Filtering.ExcludeAttributes = nil

	rules := map[string]types.Label{
		"John Doe": types.LabelPerson,
		"Jane Doe": types.LabelPerson,
	}
	proc, err := newTestProcessor(cfg, &ruleDetector{rules: rules})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	sink := new(consumertest.MetricsSink)
	proc.nextMetrics = sink

	md := pmetric.NewMetrics()
	rm := md.ResourceMetrics().AppendEmpty()
	sm := rm.ScopeMetrics().AppendEmpty()
	m := sm.Metrics().AppendEmpty()
	m.SetName("http.requests")
	m.SetEmptySum().SetIsMonotonic(true)

	ts := pcommon.NewTimestampFromTime(time.Unix(1000, 0))

	// DP 1: John Doe, 10
	dp1 := m.Sum().DataPoints().AppendEmpty()
	dp1.SetTimestamp(ts)
	dp1.Attributes().PutStr("user", "John Doe")
	dp1.Attributes().PutStr("method", "GET")
	dp1.SetIntValue(10)

	// DP 2: Jane Doe, 25
	dp2 := m.Sum().DataPoints().AppendEmpty()
	dp2.SetTimestamp(ts)
	dp2.Attributes().PutStr("user", "Jane Doe")
	dp2.Attributes().PutStr("method", "GET")
	dp2.SetIntValue(25)

	err = proc.ConsumeMetrics(context.Background(), md)
	require.NoError(t, err)

	require.Len(t, sink.AllMetrics(), 1)
	outM := sink.AllMetrics()[0].ResourceMetrics().At(0).ScopeMetrics().At(0).Metrics().At(0)
	// Colliding series merged into 1 datapoint
	require.Equal(t, 1, outM.Sum().DataPoints().Len())
	outDp := outM.Sum().DataPoints().At(0)
	assert.Equal(t, int64(35), outDp.IntValue())
	userVal, ok := outDp.Attributes().Get("user")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_PERSON]", userVal.Str())
}

func TestProcessor_FailurePolicy(t *testing.T) {
	// 1. Single invalid UTF-8 attribute under drop is scrubbed in-place and returns nil error
	cfgDrop := createDefaultConfig().(*Config)
	cfgDrop.Inference.CoalesceWindow = 5 * time.Millisecond
	cfgDrop.OnError = types.ErrorModeDrop

	procDrop, err := newTestProcessor(cfgDrop, &ruleDetector{})
	require.NoError(t, err)
	defer func() { _ = procDrop.Shutdown(context.Background()) }()

	sinkDrop := new(consumertest.TracesSink)
	procDrop.nextTraces = sinkDrop

	td := ptrace.NewTraces()
	span := td.ResourceSpans().AppendEmpty().ScopeSpans().AppendEmpty().Spans().AppendEmpty()
	span.SetName("valid_span")
	span.Attributes().PutStr("good_attr", "good_val")
	span.Attributes().PutStr("bad_attr", "bad\xffutf8")

	err = procDrop.ConsumeTraces(context.Background(), td)
	assert.NoError(t, err)

	require.Len(t, sinkDrop.AllTraces(), 1)
	outSpan := sinkDrop.AllTraces()[0].ResourceSpans().At(0).ScopeSpans().At(0).Spans().At(0)
	assert.Equal(t, "valid_span", outSpan.Name())
	goodVal, ok := outSpan.Attributes().Get("good_attr")
	require.True(t, ok)
	assert.Equal(t, "good_val", goodVal.Str())
	badVal, ok := outSpan.Attributes().Get("bad_attr")
	require.True(t, ok)
	assert.Equal(t, types.RedactedUninspected, badVal.Str())

	// 2. Systemic detector timeout under drop returns retryable (non-permanent) error
	cfgErr := createDefaultConfig().(*Config)
	cfgErr.Inference.CoalesceWindow = 5 * time.Millisecond
	cfgErr.OnError = types.ErrorModeDrop

	errDet := &ruleDetector{errToReturn: errors.New("detector timeout")}
	procErr, err := newTestProcessor(cfgErr, errDet)
	require.NoError(t, err)
	defer func() { _ = procErr.Shutdown(context.Background()) }()

	ld := plog.NewLogs()
	rec := ld.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
	rec.Body().SetStr("log payload")

	err = procErr.ConsumeLogs(context.Background(), ld)
	require.Error(t, err)
	assert.False(t, consumererror.IsPermanent(err), "systemic failure must be retryable")

	// 3. Dry run leaves payload unmodified
	cfgDry := createDefaultConfig().(*Config)
	cfgDry.Inference.CoalesceWindow = 5 * time.Millisecond
	cfgDry.DryRun = true
	rules := map[string]types.Label{"secret_password": types.LabelPerson}
	procDry, err := newTestProcessor(cfgDry, &ruleDetector{rules: rules})
	require.NoError(t, err)
	defer func() { _ = procDry.Shutdown(context.Background()) }()

	sinkDry := new(consumertest.LogsSink)
	procDry.nextLogs = sinkDry

	ldDry := plog.NewLogs()
	recDry := ldDry.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
	recDry.Body().SetStr("secret_password")

	err = procDry.ConsumeLogs(context.Background(), ldDry)
	assert.NoError(t, err)
	require.Len(t, sinkDry.AllLogs(), 1)
	assert.Equal(t, "secret_password", sinkDry.AllLogs()[0].ResourceLogs().At(0).ScopeLogs().At(0).LogRecords().At(0).Body().Str())
}

func TestProcessor_ComponentLifecycleAndEmptySignals(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.WarmupOnStart = true
	det := &ruleDetector{}
	proc, err := newTestProcessor(cfg, det)
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	// Start & Capabilities
	assert.NoError(t, proc.Start(context.Background(), nil))
	assert.True(t, proc.Capabilities().MutatesData)

	// Empty signals
	sinkLogs := new(consumertest.LogsSink)
	proc.nextLogs = sinkLogs
	assert.NoError(t, proc.ConsumeLogs(context.Background(), plog.NewLogs()))
	assert.Len(t, sinkLogs.AllLogs(), 1)

	sinkTraces := new(consumertest.TracesSink)
	proc.nextTraces = sinkTraces
	assert.NoError(t, proc.ConsumeTraces(context.Background(), ptrace.NewTraces()))
	assert.Len(t, sinkTraces.AllTraces(), 1)

	sinkMetrics := new(consumertest.MetricsSink)
	proc.nextMetrics = sinkMetrics
	assert.NoError(t, proc.ConsumeMetrics(context.Background(), pmetric.NewMetrics()))
	assert.Len(t, sinkMetrics.AllMetrics(), 1)
}

func TestProcessor_LogBodyTypes(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	rules := map[string]types.Label{
		"John Doe": types.LabelPerson,
	}
	proc, err := newTestProcessor(cfg, &ruleDetector{rules: rules})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	sink := new(consumertest.LogsSink)
	proc.nextLogs = sink

	ld := plog.NewLogs()
	recSlice := ld.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
	// Slice body
	recSlice.Body().SetEmptySlice()
	recSlice.Body().Slice().AppendEmpty().SetStr("John Doe")

	// Bytes body (skipped)
	recBytes := ld.ResourceLogs().At(0).ScopeLogs().At(0).LogRecords().AppendEmpty()
	recBytes.Body().SetEmptyBytes().FromRaw([]byte{0x01, 0x02, 0x03})

	err = proc.ConsumeLogs(context.Background(), ld)
	require.NoError(t, err)

	outRecSlice := sink.AllLogs()[0].ResourceLogs().At(0).ScopeLogs().At(0).LogRecords().At(0)
	assert.Equal(t, "[REDACTED_PERSON]", outRecSlice.Body().Slice().At(0).Str())
}

func TestProcessor_AllMetricDataPointTypes(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	rules := map[string]types.Label{
		"John Doe": types.LabelPerson,
	}
	proc, err := newTestProcessor(cfg, &ruleDetector{rules: rules})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	sink := new(consumertest.MetricsSink)
	proc.nextMetrics = sink

	md := pmetric.NewMetrics()
	sm := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty()

	// 1. Gauge
	mGauge := sm.Metrics().AppendEmpty()
	mGauge.SetName("metric.gauge")
	dpGauge := mGauge.SetEmptyGauge().DataPoints().AppendEmpty()
	dpGauge.Attributes().PutStr("user", "John Doe")

	// 2. Histogram
	mHist := sm.Metrics().AppendEmpty()
	mHist.SetName("metric.histogram")
	dpHist := mHist.SetEmptyHistogram().DataPoints().AppendEmpty()
	dpHist.Attributes().PutStr("user", "John Doe")

	// 3. ExponentialHistogram
	mExpHist := sm.Metrics().AppendEmpty()
	mExpHist.SetName("metric.exphist")
	dpExpHist := mExpHist.SetEmptyExponentialHistogram().DataPoints().AppendEmpty()
	dpExpHist.Attributes().PutStr("user", "John Doe")

	// 4. Summary
	mSummary := sm.Metrics().AppendEmpty()
	mSummary.SetName("metric.summary")
	dpSummary := mSummary.SetEmptySummary().DataPoints().AppendEmpty()
	dpSummary.Attributes().PutStr("user", "John Doe")

	err = proc.ConsumeMetrics(context.Background(), md)
	require.NoError(t, err)

	outM := sink.AllMetrics()[0].ResourceMetrics().At(0).ScopeMetrics().At(0).Metrics()
	require.Equal(t, 4, outM.Len())

	v1, _ := outM.At(0).Gauge().DataPoints().At(0).Attributes().Get("user")
	assert.Equal(t, "[REDACTED_PERSON]", v1.Str())

	v2, _ := outM.At(1).Histogram().DataPoints().At(0).Attributes().Get("user")
	assert.Equal(t, "[REDACTED_PERSON]", v2.Str())

	v3, _ := outM.At(2).ExponentialHistogram().DataPoints().At(0).Attributes().Get("user")
	assert.Equal(t, "[REDACTED_PERSON]", v3.Str())

	v4, _ := outM.At(3).Summary().DataPoints().At(0).Attributes().Get("user")
	assert.Equal(t, "[REDACTED_PERSON]", v4.Str())
}

func TestProcessor_TelemetryHelpers(t *testing.T) {
	set := processortest.NewNopSettings(Type)
	tel, err := newTelemetry(&set.TelemetrySettings, "1.31.0", "sha256", "mlas")
	require.NoError(t, err)

	tel.RecordSkipped("reason", 1)
	tel.RecordDropped("logs", "unparseable", 1)
	tel.RecordMetricLabelPII("metric.name", 1)
	tel.RecordInspected("logs", 1)
	tel.RecordError("err")
	tel.RecordInflight(1)
	tel.RecordInflight(-1)
	tel.RecordQueueWait(5 * time.Millisecond)
	assert.NoError(t, tel.Close())

	// Nil receiver safe
	var nilTel *telemetry
	nilTel.RecordSkipped("reason", 1)
	nilTel.RecordDropped("logs", "unparseable", 1)
	nilTel.RecordMetricLabelPII("metric.name", 1)
	nilTel.RecordInspected("logs", 1)
	nilTel.RecordError("err")
	nilTel.RecordBatch(1)
	nilTel.RecordTensorBatch(1, 1.0)
	nilTel.RecordStageDuration("stage", time.Millisecond)
	nilTel.RecordEntity(types.LabelPerson)
	nilTel.RecordScrubbed("signal", "reason")
	nilTel.RecordMetricCollision("Sum", 1)
	nilTel.RecordInflight(1)
	nilTel.RecordQueueWait(time.Millisecond)
	assert.NoError(t, nilTel.Close())
}

type errWarmupDetector struct {
	detect.Detector
}

func (e *errWarmupDetector) Warmup(_ context.Context) error {
	return errors.New("warmup failed")
}
func (e *errWarmupDetector) Close() error { return nil }

func TestProcessor_NewProcessorErrorsAndWarmupFailure(t *testing.T) {
	set := processortest.NewNopSettings(Type)

	// nil config
	_, err := newProcessor(nil, &set.TelemetrySettings, nil, nil)
	assert.ErrorContains(t, err, "config is nil")

	// missing tokenizer path
	cfg := createDefaultConfig().(*Config)
	cfg.Model.TokenizerPath = ""
	_, err = newProcessor(cfg, &set.TelemetrySettings, nil, nil)
	assert.ErrorContains(t, err, "model.tokenizer_path is required")

	// Warmup failure in Start
	proc, err := newTestProcessor(cfg, &errWarmupDetector{})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()
	err = proc.Start(context.Background(), nil)
	assert.ErrorContains(t, err, "warmup failed")
}

func TestProcessor_RealONNXEngineAndEntitiesConfig(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Model.Path = "testdata/tiny_ner.onnx"
	cfg.Model.TokenizerPath = "testdata/tokenizer.json"
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	cfg.Filtering.MinStringLength = 1
	cfg.Filtering.ExcludeAttributes = nil
	cfg.Entities = []string{"EMAIL"}

	set := processortest.NewNopSettings(Type)
	proc, err := newProcessor(cfg, &set.TelemetrySettings, nil, nil)
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	sink := new(consumertest.LogsSink)
	proc.nextLogs = sink

	ld := plog.NewLogs()
	rec := ld.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
	rec.Body().SetStr("tokenbperson tokenbemail tokenbperson-tokenbemail")

	err = proc.ConsumeLogs(context.Background(), ld)
	require.NoError(t, err)

	require.Len(t, sink.AllLogs(), 1)
	outRec := sink.AllLogs()[0].ResourceLogs().At(0).ScopeLogs().At(0).LogRecords().At(0)
	assert.Equal(t, "tokenbperson [REDACTED_EMAIL] tokenbperson-[REDACTED_EMAIL]", outRec.Body().Str())
}

func TestProcessor_SpanStatusAndMetricExemplarsAndMultiTimestampGauge(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	cfg.Filtering.MinStringLength = 1
	cfg.Filtering.ExcludeAttributes = nil

	rules := map[string]types.Label{
		"alice@example.com": types.LabelEmail,
		"John Doe":          types.LabelPerson,
	}
	proc, err := newTestProcessor(cfg, &ruleDetector{rules: rules})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	// 1. Span Status Message
	sinkTraces := new(consumertest.TracesSink)
	proc.nextTraces = sinkTraces

	td := ptrace.NewTraces()
	span := td.ResourceSpans().AppendEmpty().ScopeSpans().AppendEmpty().Spans().AppendEmpty()
	span.SetName("op")
	span.Status().SetMessage("error from alice@example.com")

	err = proc.ConsumeTraces(context.Background(), td)
	require.NoError(t, err)
	require.Len(t, sinkTraces.AllTraces(), 1)
	outSpan := sinkTraces.AllTraces()[0].ResourceSpans().At(0).ScopeSpans().At(0).Spans().At(0)
	assert.Equal(t, "error from [REDACTED_EMAIL]", outSpan.Status().Message())

	// 2. Metric Exemplars & Multi-timestamp Gauge preservation
	sinkMetrics := new(consumertest.MetricsSink)
	proc.nextMetrics = sinkMetrics

	md := pmetric.NewMetrics()
	sm := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty()

	// Sum with Exemplar
	mSum := sm.Metrics().AppendEmpty()
	mSum.SetName("http.requests")
	dpSum := mSum.SetEmptySum().DataPoints().AppendEmpty()
	ex := dpSum.Exemplars().AppendEmpty()
	ex.FilteredAttributes().PutStr("user", "John Doe")

	// Gauge with 2 datapoints at ts1 and ts2 having clean attributes (host: "srv1")
	ts1 := pcommon.NewTimestampFromTime(time.Unix(1000, 0))
	ts2 := pcommon.NewTimestampFromTime(time.Unix(2000, 0))
	mGauge := sm.Metrics().AppendEmpty()
	mGauge.SetName("cpu.load")
	dp1 := mGauge.SetEmptyGauge().DataPoints().AppendEmpty()
	dp1.SetTimestamp(ts1)
	dp1.Attributes().PutStr("host", "srv1")
	dp1.SetDoubleValue(10.5)

	dp2 := mGauge.Gauge().DataPoints().AppendEmpty()
	dp2.SetTimestamp(ts2)
	dp2.Attributes().PutStr("host", "srv1")
	dp2.SetDoubleValue(20.5)

	err = proc.ConsumeMetrics(context.Background(), md)
	require.NoError(t, err)

	require.Len(t, sinkMetrics.AllMetrics(), 1)
	outMetrics := sinkMetrics.AllMetrics()[0].ResourceMetrics().At(0).ScopeMetrics().At(0).Metrics()
	require.Equal(t, 2, outMetrics.Len())

	outEx := outMetrics.At(0).Sum().DataPoints().At(0).Exemplars().At(0)
	userVal, ok := outEx.FilteredAttributes().Get("user")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_PERSON]", userVal.Str())

	require.Equal(t, 2, outMetrics.At(1).Gauge().DataPoints().Len())
}

func TestProcessor_MaxNestingDepth(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	cfg.OnError = types.ErrorModeDrop

	proc, err := newTestProcessor(cfg, &ruleDetector{})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	sink := new(consumertest.LogsSink)
	proc.nextLogs = sink

	ld := plog.NewLogs()
	rec := ld.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()

	curr := rec.Body().SetEmptyMap()
	for i := 0; i < 34; i++ {
		curr = curr.PutEmptyMap(fmt.Sprintf("k%d", i))
	}
	curr.PutStr("secret_key", "secret")

	err = proc.ConsumeLogs(context.Background(), ld)
	require.NoError(t, err)

	require.Len(t, sink.AllLogs(), 1)
	outCurr := sink.AllLogs()[0].ResourceLogs().At(0).ScopeLogs().At(0).LogRecords().At(0).Body().Map()
	for i := 0; i < 34; i++ {
		v, ok := outCurr.Get(fmt.Sprintf("k%d", i))
		require.True(t, ok)
		outCurr = v.Map()
	}
	leaf, ok := outCurr.Get("secret_key")
	require.True(t, ok)
	assert.Equal(t, types.RedactedUninspected, leaf.Str())

	// Dry-run mode with on_error: drop must NOT scrub deeply nested values,
	// but MUST still record redaction_skipped_total{reason="max_depth"}
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	defer func() { _ = mp.Shutdown(context.Background()) }()

	cfgDry := createDefaultConfig().(*Config)
	cfgDry.Inference.CoalesceWindow = 5 * time.Millisecond
	cfgDry.OnError = types.ErrorModeDrop
	cfgDry.DryRun = true

	setDry := processortest.NewNopSettings(Type)
	setDry.MeterProvider = mp
	procDry, err := newProcessor(cfgDry, &setDry.TelemetrySettings, &mockTokenizer{}, &ruleDetector{})
	require.NoError(t, err)
	defer func() { _ = procDry.Shutdown(context.Background()) }()

	sinkDry := new(consumertest.LogsSink)
	procDry.nextLogs = sinkDry

	ldDry := plog.NewLogs()
	recDry := ldDry.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
	currDry := recDry.Body().SetEmptyMap()
	for i := 0; i < 34; i++ {
		currDry = currDry.PutEmptyMap(fmt.Sprintf("k%d", i))
	}
	currDry.PutStr("secret_key", "secret")

	require.NoError(t, procDry.ConsumeLogs(context.Background(), ldDry))
	require.Len(t, sinkDry.AllLogs(), 1)
	outDry := sinkDry.AllLogs()[0].ResourceLogs().At(0).ScopeLogs().At(0).LogRecords().At(0).Body().Map()
	for i := 0; i < 34; i++ {
		v, ok := outDry.Get(fmt.Sprintf("k%d", i))
		require.True(t, ok)
		outDry = v.Map()
	}
	leafDry, ok := outDry.Get("secret_key")
	require.True(t, ok)
	assert.Equal(t, "secret", leafDry.Str(), "dry_run must preserve deeply nested string")

	var rm metricdata.ResourceMetrics
	require.NoError(t, reader.Collect(context.Background(), &rm))
	var skippedCount, scrubbedCount int64
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			switch m.Name {
			case "redaction_skipped_total":
				for _, dp := range m.Data.(metricdata.Sum[int64]).DataPoints {
					skippedCount += dp.Value
				}
			case "redaction_scrubbed_total":
				for _, dp := range m.Data.(metricdata.Sum[int64]).DataPoints {
					scrubbedCount += dp.Value
				}
			}
		}
	}
	assert.Equal(t, int64(1), skippedCount, "redaction_skipped_total must be recorded in dry_run")
	assert.Equal(t, int64(0), scrubbedCount, "redaction_scrubbed_total must be 0 in dry_run")
}

type closeTrackingTokenizer struct {
	tokenizer.Tokenizer
	closed int
}

func (c *closeTrackingTokenizer) Close() error {
	c.closed++
	return nil
}

func TestProcessor_TokenizerCloseOnShutdownAndInitError(t *testing.T) {
	cfg := createDefaultConfig().(*Config)
	cfg.Model.TokenizerPath = filepath.Clean("testdata/tokenizer.json")
	cfg.Model.Path = "testdata/nonexistent_model.onnx"

	baseTok, err := tokenizer.NewWordPieceTokenizer(cfg.Model.TokenizerPath, "")
	require.NoError(t, err)
	trackTok := &closeTrackingTokenizer{Tokenizer: baseTok}

	set := processortest.NewNopSettings(Type)
	// 1. When detector init fails, newProcessor must close tokenizer and telemetry
	_, err = newProcessor(cfg, &set.TelemetrySettings, trackTok, nil)
	require.Error(t, err)
	assert.Equal(t, 1, trackTok.closed, "tokenizer must be closed when newProcessor fails during engine init")

	// 2. When processor succeeds and Shutdown is called, tokenizer.Close() must be called
	trackTok2 := &closeTrackingTokenizer{Tokenizer: baseTok}
	proc, err := newProcessor(cfg, &set.TelemetrySettings, trackTok2, &ruleDetector{})
	require.NoError(t, err)
	require.NoError(t, proc.Shutdown(context.Background()))
	assert.Equal(t, 1, trackTok2.closed, "tokenizer must be closed on processor Shutdown")
}
