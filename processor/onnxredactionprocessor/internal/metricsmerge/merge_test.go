package metricsmerge

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/pmetric"
)

func TestComputeCanonicalAttrHash(t *testing.T) {
	attrs1 := pcommon.NewMap()
	attrs1.PutStr("user", "[REDACTED_PERSON]")
	attrs1.PutStr("env", "prod")

	attrs2 := pcommon.NewMap()
	attrs2.PutStr("env", "prod")
	attrs2.PutStr("user", "[REDACTED_PERSON]")

	var scratch []string
	h1, scratch := ComputeCanonicalAttrHash(attrs1, scratch)
	h2, _ := ComputeCanonicalAttrHash(attrs2, scratch)

	assert.Equal(t, h1, h2, "maps with same keys/values in different order must yield identical hash")

	attrs3 := pcommon.NewMap()
	attrs3.PutStr("user", "other")
	h3, _ := ComputeCanonicalAttrHash(attrs3, scratch)
	assert.NotEqual(t, h1, h3)
}

func TestResolveCollisions_Sum(t *testing.T) {
	md := pmetric.NewMetrics()
	rm := md.ResourceMetrics().AppendEmpty()
	sm := rm.ScopeMetrics().AppendEmpty()
	m := sm.Metrics().AppendEmpty()
	m.SetName("http.requests.total")
	sum := m.SetEmptySum()
	sum.SetIsMonotonic(true)

	ts := pcommon.Timestamp(1000000000)

	// Datapoint 1: val 10
	dp1 := sum.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("user", "[REDACTED_PERSON]")
	dp1.SetTimestamp(ts)
	dp1.SetIntValue(10)

	// Datapoint 2: val 25, same user attribute and timestamp
	dp2 := sum.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("user", "[REDACTED_PERSON]")
	dp2.SetTimestamp(ts)
	dp2.SetIntValue(25)

	stats := ResolveCollisions(m)
	assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeSum])

	// Must be merged into 1 datapoint with value 35
	require.Equal(t, 1, sum.DataPoints().Len())
	mergedDP := sum.DataPoints().At(0)
	assert.Equal(t, int64(35), mergedDP.IntValue())
	userVal, ok := mergedDP.Attributes().Get("user")
	require.True(t, ok)
	assert.Equal(t, "[REDACTED_PERSON]", userVal.Str())
}

func TestResolveCollisions_Gauge(t *testing.T) {
	md := pmetric.NewMetrics()
	m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
	m.SetName("system.memory.usage")
	gauge := m.SetEmptyGauge()

	ts1 := pcommon.Timestamp(1000000000)
	ts2 := pcommon.Timestamp(2000000000) // later timestamp

	dp1 := gauge.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("host", "srv1")
	dp1.SetTimestamp(ts1)
	dp1.SetDoubleValue(50.0)

	dp2 := gauge.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("host", "srv1")
	dp2.SetTimestamp(ts2)
	dp2.SetDoubleValue(85.0)

	stats := ResolveCollisions(m)
	assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeGauge])

	require.Equal(t, 1, gauge.DataPoints().Len())
	// Should keep latest timestamp
	assert.Equal(t, 85.0, gauge.DataPoints().At(0).DoubleValue())
	assert.Equal(t, ts2, gauge.DataPoints().At(0).Timestamp())
}

func TestResolveCollisions_Histogram(t *testing.T) {
	md := pmetric.NewMetrics()
	m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
	m.SetName("http.server.duration")
	hist := m.SetEmptyHistogram()

	ts := pcommon.Timestamp(1000000000)
	bounds := []float64{0.1, 0.5, 1.0}

	dp1 := hist.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("user", "[REDACTED_PATH_USER]")
	dp1.SetTimestamp(ts)
	dp1.ExplicitBounds().FromRaw(bounds)
	dp1.SetCount(10)
	dp1.SetSum(5.0)
	dp1.SetMin(0.05)
	dp1.SetMax(0.8)
	dp1.BucketCounts().FromRaw([]uint64{2, 5, 3, 0})

	dp2 := hist.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("user", "[REDACTED_PATH_USER]")
	dp2.SetTimestamp(ts)
	dp2.ExplicitBounds().FromRaw(bounds)
	dp2.SetCount(15)
	dp2.SetSum(12.0)
	dp2.SetMin(0.02)
	dp2.SetMax(0.95)
	dp2.BucketCounts().FromRaw([]uint64{3, 7, 4, 1})

	stats := ResolveCollisions(m)
	assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeHistogram])

	require.Equal(t, 1, hist.DataPoints().Len())
	dpMerged := hist.DataPoints().At(0)
	assert.Equal(t, uint64(25), dpMerged.Count())
	assert.Equal(t, 17.0, dpMerged.Sum())
	assert.Equal(t, 0.02, dpMerged.Min())
	assert.Equal(t, 0.95, dpMerged.Max())
	assert.Equal(t, []uint64{5, 12, 7, 1}, dpMerged.BucketCounts().AsRaw())
}

func TestResolveCollisions_Summary(t *testing.T) {
	md := pmetric.NewMetrics()
	m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
	m.SetName("request.summary")
	summary := m.SetEmptySummary()

	ts := pcommon.Timestamp(1000000000)

	dp1 := summary.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("user", "[REDACTED_PERSON]")
	dp1.SetTimestamp(ts)

	dp2 := summary.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("user", "[REDACTED_PERSON]")
	dp2.SetTimestamp(ts)

	stats := ResolveCollisions(m)
	assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeSummary])
	require.Equal(t, 1, summary.DataPoints().Len())
}

func TestResolveCollisions_ExponentialHistogram(t *testing.T) {
	md := pmetric.NewMetrics()
	m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
	m.SetName("exp.hist")
	eh := m.SetEmptyExponentialHistogram()

	ts := pcommon.Timestamp(1000000000)

	dp1 := eh.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("user", "[REDACTED_PERSON]")
	dp1.SetTimestamp(ts)
	dp1.SetCount(10)
	dp1.SetSum(100.0)
	dp1.SetMin(5.0)
	dp1.SetMax(20.0)
	dp1.SetZeroCount(2)
	dp1.Positive().SetOffset(1)
	dp1.Positive().BucketCounts().FromRaw([]uint64{3, 5})
	dp1.Negative().SetOffset(0)
	dp1.Negative().BucketCounts().FromRaw([]uint64{1})

	dp2 := eh.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("user", "[REDACTED_PERSON]")
	dp2.SetTimestamp(ts)
	dp2.SetCount(5)
	dp2.SetSum(40.0)
	dp2.SetMin(2.0)
	dp2.SetMax(15.0)
	dp2.SetZeroCount(1)
	dp2.Positive().SetOffset(1)
	dp2.Positive().BucketCounts().FromRaw([]uint64{2, 2})
	dp2.Negative().SetOffset(0)
	dp2.Negative().BucketCounts().FromRaw([]uint64{4})

	stats := ResolveCollisions(m)
	assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeExponentialHistogram])
	require.Equal(t, 1, eh.DataPoints().Len())
	dpMerged := eh.DataPoints().At(0)
	assert.Equal(t, uint64(15), dpMerged.Count())
	assert.Equal(t, 140.0, dpMerged.Sum())
	assert.Equal(t, 2.0, dpMerged.Min())
	assert.Equal(t, 20.0, dpMerged.Max())
	assert.Equal(t, uint64(3), dpMerged.ZeroCount())
	assert.Equal(t, []uint64{5, 7}, dpMerged.Positive().BucketCounts().AsRaw())
	assert.Equal(t, []uint64{5}, dpMerged.Negative().BucketCounts().AsRaw())
}

func TestResolveCollisions_ExponentialHistogram_OffsetAndScale(t *testing.T) {
	t.Run("offset expansion and empty buckets", func(t *testing.T) {
		md := pmetric.NewMetrics()
		m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
		m.SetName("exp.hist.offset")
		eh := m.SetEmptyExponentialHistogram()

		ts := pcommon.Timestamp(1000000000)

		dp1 := eh.DataPoints().AppendEmpty()
		dp1.Attributes().PutStr("k", "v")
		dp1.SetTimestamp(ts)
		dp1.Positive().SetOffset(2)
		dp1.Positive().BucketCounts().FromRaw([]uint64{10, 20}) // indices 2, 3
		// dp1 negative left empty

		dp2 := eh.DataPoints().AppendEmpty()
		dp2.Attributes().PutStr("k", "v")
		dp2.SetTimestamp(ts)
		dp2.Positive().SetOffset(0)
		dp2.Positive().BucketCounts().FromRaw([]uint64{1, 2, 3}) // indices 0, 1, 2
		dp2.Negative().SetOffset(-1)
		dp2.Negative().BucketCounts().FromRaw([]uint64{42}) // index -1

		stats := ResolveCollisions(m)
		assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeExponentialHistogram])
		require.Equal(t, 1, eh.DataPoints().Len())
		dpMerged := eh.DataPoints().At(0)

		// Positive should have range [0, 4) -> indices 0, 1, 2, 3
		assert.Equal(t, int32(0), dpMerged.Positive().Offset())
		assert.Equal(t, []uint64{1, 2, 13, 20}, dpMerged.Positive().BucketCounts().AsRaw())

		// Negative was empty in dp1, populated from dp2
		assert.Equal(t, int32(-1), dpMerged.Negative().Offset())
		assert.Equal(t, []uint64{42}, dpMerged.Negative().BucketCounts().AsRaw())
	})

	t.Run("scale mismatch preserves buckets and merges zero count", func(t *testing.T) {
		md := pmetric.NewMetrics()
		m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
		m.SetName("exp.hist.scale_mismatch")
		eh := m.SetEmptyExponentialHistogram()

		ts := pcommon.Timestamp(1000000000)

		dp1 := eh.DataPoints().AppendEmpty()
		dp1.Attributes().PutStr("k", "v")
		dp1.SetTimestamp(ts)
		dp1.SetScale(2)
		dp1.SetZeroCount(5)
		dp1.Positive().SetOffset(1)
		dp1.Positive().BucketCounts().FromRaw([]uint64{10})

		dp2 := eh.DataPoints().AppendEmpty()
		dp2.Attributes().PutStr("k", "v")
		dp2.SetTimestamp(ts)
		dp2.SetScale(3) // different scale
		dp2.SetZeroCount(7)
		dp2.Positive().SetOffset(1)
		dp2.Positive().BucketCounts().FromRaw([]uint64{20})

		stats := ResolveCollisions(m)
		assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeExponentialHistogram])
		require.Equal(t, 1, eh.DataPoints().Len())
		dpMerged := eh.DataPoints().At(0)

		assert.Equal(t, uint64(12), dpMerged.ZeroCount())
		assert.Equal(t, int32(2), dpMerged.Scale())
		// Buckets must not be merged due to scale mismatch
		assert.Equal(t, []uint64{10}, dpMerged.Positive().BucketCounts().AsRaw())
	})
}

func TestResolveCollisions_SumDoubleAndMixed(t *testing.T) {
	md := pmetric.NewMetrics()
	m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
	sum := m.SetEmptySum()

	ts := pcommon.Timestamp(1000000000)

	// Int + Double collision
	dp1 := sum.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("key", "val")
	dp1.SetTimestamp(ts)
	dp1.SetIntValue(10)

	dp2 := sum.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("key", "val")
	dp2.SetTimestamp(ts)
	dp2.SetDoubleValue(2.5)

	stats := ResolveCollisions(m)
	assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeSum])
	require.Equal(t, 1, sum.DataPoints().Len())
	assert.Equal(t, 12.5, sum.DataPoints().At(0).DoubleValue())

	// Test single datapoint short-circuit
	stats2 := ResolveCollisions(m)
	assert.Empty(t, stats2.CollisionsByType)

	// Empty metric
	mEmpty := pmetric.NewMetric()
	statsEmpty := ResolveCollisions(mEmpty)
	assert.Empty(t, statsEmpty.CollisionsByType)
}

func TestResolveCollisions_GaugeOlderTimestamp(t *testing.T) {
	md := pmetric.NewMetrics()
	m := md.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
	gauge := m.SetEmptyGauge()

	ts1 := pcommon.Timestamp(2000000000)
	ts2 := pcommon.Timestamp(1000000000) // older timestamp

	dp1 := gauge.DataPoints().AppendEmpty()
	dp1.Attributes().PutStr("host", "srv1")
	dp1.SetTimestamp(ts1)
	dp1.SetDoubleValue(100.0)

	dp2 := gauge.DataPoints().AppendEmpty()
	dp2.Attributes().PutStr("host", "srv1")
	dp2.SetTimestamp(ts2)
	dp2.SetDoubleValue(50.0)

	stats := ResolveCollisions(m)
	assert.Equal(t, 1, stats.CollisionsByType[pmetric.MetricTypeGauge])
	require.Equal(t, 1, gauge.DataPoints().Len())
	// Should keep first datapoint with ts1
	assert.Equal(t, 100.0, gauge.DataPoints().At(0).DoubleValue())
	assert.Equal(t, ts1, gauge.DataPoints().At(0).Timestamp())
}
