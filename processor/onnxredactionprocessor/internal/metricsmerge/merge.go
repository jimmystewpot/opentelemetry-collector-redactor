package metricsmerge

import (
	"sort"

	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/pmetric"
)

// CollisionStats records the number of metric series collisions resolved per metric type.
type CollisionStats struct {
	CollisionsByType map[pmetric.MetricType]int
}

const (
	fnvOffset64 = 14695981039346656037
	fnvPrime64  = 1099511628211
)

// ComputeCanonicalAttrHash computes an allocation-free 64-bit FNV-64a hash over pcommon.Map using keyScratch.
func ComputeCanonicalAttrHash(attrs pcommon.Map, keyScratch []string) (uint64, []string) {
	if attrs.Len() == 0 {
		return fnvOffset64, keyScratch
	}

	keyScratch = keyScratch[:0]
	attrs.Range(func(k string, _ pcommon.Value) bool {
		keyScratch = append(keyScratch, k)
		return true
	})
	sort.Strings(keyScratch)

	h := uint64(fnvOffset64)
	for _, k := range keyScratch {
		for i := 0; i < len(k); i++ {
			h ^= uint64(k[i])
			h *= fnvPrime64
		}
		h ^= '='
		h *= fnvPrime64

		if v, ok := attrs.Get(k); ok {
			s := v.AsString()
			for i := 0; i < len(s); i++ {
				h ^= uint64(s[i])
				h *= fnvPrime64
			}
		}
		h ^= ';'
		h *= fnvPrime64
	}

	return h, keyScratch
}

type seriesKey struct {
	attrHash uint64
	startTS  pcommon.Timestamp
	ts       pcommon.Timestamp
}

func (s *CollisionStats) record(t pmetric.MetricType) {
	if s.CollisionsByType == nil {
		s.CollisionsByType = make(map[pmetric.MetricType]int)
	}
	s.CollisionsByType[t]++
}

// ResolveCollisions merges duplicate datapoints resulting from attribute redaction within a metric.
func ResolveCollisions(m pmetric.Metric) CollisionStats {
	var stats CollisionStats
	var keyScratch []string

	switch m.Type() {
	case pmetric.MetricTypeSum:
		dps := m.Sum().DataPoints()
		if dps.Len() <= 1 {
			return stats
		}

		merged := pmetric.NewNumberDataPointSlice()
		indices := make(map[seriesKey]int)

		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			h, ks := ComputeCanonicalAttrHash(dp.Attributes(), keyScratch)
			keyScratch = ks
			k := seriesKey{attrHash: h, startTS: dp.StartTimestamp(), ts: dp.Timestamp()}

			if existingIdx, found := indices[k]; found {
				target := merged.At(existingIdx)
				if target.ValueType() == pmetric.NumberDataPointValueTypeInt && dp.ValueType() == pmetric.NumberDataPointValueTypeInt {
					target.SetIntValue(target.IntValue() + dp.IntValue())
				} else {
					var v1, v2 float64
					if target.ValueType() == pmetric.NumberDataPointValueTypeInt {
						v1 = float64(target.IntValue())
					} else {
						v1 = target.DoubleValue()
					}
					if dp.ValueType() == pmetric.NumberDataPointValueTypeInt {
						v2 = float64(dp.IntValue())
					} else {
						v2 = dp.DoubleValue()
					}
					target.SetDoubleValue(v1 + v2)
				}
				stats.record(pmetric.MetricTypeSum)
			} else {
				indices[k] = merged.Len()
				newDP := merged.AppendEmpty()
				dp.CopyTo(newDP)
			}
		}
		merged.CopyTo(dps)

	case pmetric.MetricTypeGauge:
		dps := m.Gauge().DataPoints()
		if dps.Len() <= 1 {
			return stats
		}

		merged := pmetric.NewNumberDataPointSlice()
		indices := make(map[uint64]int)

		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			h, ks := ComputeCanonicalAttrHash(dp.Attributes(), keyScratch)
			keyScratch = ks

			if existingIdx, found := indices[h]; found {
				target := merged.At(existingIdx)
				// Gauge keeps latest timestamp
				if dp.Timestamp() >= target.Timestamp() {
					dp.CopyTo(target)
				}
				stats.record(pmetric.MetricTypeGauge)
			} else {
				indices[h] = merged.Len()
				newDP := merged.AppendEmpty()
				dp.CopyTo(newDP)
			}
		}
		merged.CopyTo(dps)

	case pmetric.MetricTypeHistogram:
		dps := m.Histogram().DataPoints()
		if dps.Len() <= 1 {
			return stats
		}

		merged := pmetric.NewHistogramDataPointSlice()
		indices := make(map[seriesKey]int)

		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			h, ks := ComputeCanonicalAttrHash(dp.Attributes(), keyScratch)
			keyScratch = ks
			k := seriesKey{attrHash: h, startTS: dp.StartTimestamp(), ts: dp.Timestamp()}

			if existingIdx, found := indices[k]; found {
				target := merged.At(existingIdx)
				target.SetCount(target.Count() + dp.Count())
				if target.HasSum() && dp.HasSum() {
					target.SetSum(target.Sum() + dp.Sum())
				}
				if dp.HasMin() {
					if !target.HasMin() || dp.Min() < target.Min() {
						target.SetMin(dp.Min())
					}
				}
				if dp.HasMax() {
					if !target.HasMax() || dp.Max() > target.Max() {
						target.SetMax(dp.Max())
					}
				}

				if target.BucketCounts().Len() == dp.BucketCounts().Len() {
					for b := 0; b < target.BucketCounts().Len(); b++ {
						target.BucketCounts().SetAt(b, target.BucketCounts().At(b)+dp.BucketCounts().At(b))
					}
				}
				stats.record(pmetric.MetricTypeHistogram)
			} else {
				indices[k] = merged.Len()
				newDP := merged.AppendEmpty()
				dp.CopyTo(newDP)
			}
		}
		merged.CopyTo(dps)

	case pmetric.MetricTypeSummary:
		dps := m.Summary().DataPoints()
		if dps.Len() <= 1 {
			return stats
		}

		merged := pmetric.NewSummaryDataPointSlice()
		indices := make(map[seriesKey]int)

		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			h, ks := ComputeCanonicalAttrHash(dp.Attributes(), keyScratch)
			keyScratch = ks
			k := seriesKey{attrHash: h, startTS: dp.StartTimestamp(), ts: dp.Timestamp()}

			if _, found := indices[k]; found {
				stats.record(pmetric.MetricTypeSummary)
			} else {
				indices[k] = merged.Len()
				newDP := merged.AppendEmpty()
				dp.CopyTo(newDP)
			}
		}
		merged.CopyTo(dps)

	case pmetric.MetricTypeExponentialHistogram:
		dps := m.ExponentialHistogram().DataPoints()
		if dps.Len() <= 1 {
			return stats
		}

		merged := pmetric.NewExponentialHistogramDataPointSlice()
		indices := make(map[seriesKey]int)

		for i := 0; i < dps.Len(); i++ {
			dp := dps.At(i)
			h, ks := ComputeCanonicalAttrHash(dp.Attributes(), keyScratch)
			keyScratch = ks
			k := seriesKey{attrHash: h, startTS: dp.StartTimestamp(), ts: dp.Timestamp()}

			if existingIdx, found := indices[k]; found {
				target := merged.At(existingIdx)
				target.SetCount(target.Count() + dp.Count())
				if target.HasSum() && dp.HasSum() {
					target.SetSum(target.Sum() + dp.Sum())
				}
				if dp.HasMin() {
					if !target.HasMin() || dp.Min() < target.Min() {
						target.SetMin(dp.Min())
					}
				}
				if dp.HasMax() {
					if !target.HasMax() || dp.Max() > target.Max() {
						target.SetMax(dp.Max())
					}
				}
				target.SetZeroCount(target.ZeroCount() + dp.ZeroCount())
				if target.Scale() == dp.Scale() {
					mergeExpBuckets(target.Positive(), dp.Positive())
					mergeExpBuckets(target.Negative(), dp.Negative())
				}
				stats.record(pmetric.MetricTypeExponentialHistogram)
			} else {
				indices[k] = merged.Len()
				newDP := merged.AppendEmpty()
				dp.CopyTo(newDP)
			}
		}
		merged.CopyTo(dps)

	case pmetric.MetricTypeEmpty:
		// No data points to process
	}

	return stats
}

func mergeExpBuckets(dst, src pmetric.ExponentialHistogramDataPointBuckets) {
	srcLen := src.BucketCounts().Len()
	if srcLen == 0 {
		return
	}
	dstLen := dst.BucketCounts().Len()
	if dstLen == 0 {
		dst.SetOffset(src.Offset())
		dst.BucketCounts().FromRaw(append([]uint64(nil), src.BucketCounts().AsRaw()...))
		return
	}

	dstOffset := dst.Offset()
	srcOffset := src.Offset()
	minOffset := dstOffset
	if srcOffset < minOffset {
		minOffset = srcOffset
	}

	dstEnd := int(dstOffset) + dstLen
	srcEnd := int(srcOffset) + srcLen
	maxEnd := dstEnd
	if srcEnd > maxEnd {
		maxEnd = srcEnd
	}

	newLen := maxEnd - int(minOffset)
	newCounts := make([]uint64, newLen)

	dstRaw := dst.BucketCounts().AsRaw()
	dstBase := int(dstOffset) - int(minOffset)
	for i := 0; i < dstLen; i++ {
		newCounts[dstBase+i] += dstRaw[i]
	}

	srcRaw := src.BucketCounts().AsRaw()
	srcBase := int(srcOffset) - int(minOffset)
	for i := 0; i < srcLen; i++ {
		newCounts[srcBase+i] += srcRaw[i]
	}

	dst.SetOffset(minOffset)
	dst.BucketCounts().FromRaw(newCounts)
}
