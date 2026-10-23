package dedup

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/pmetric"
	"go.opentelemetry.io/collector/pdata/ptrace"
)

func TestMutationTarget_Set(t *testing.T) {
	// 1. TargetPCommonValue
	val := pcommon.NewValueStr("initial_val")
	tVal := MutationTarget{Kind: TargetPCommonValue, Val: val}
	tVal.Set("mutated_val")
	assert.Equal(t, "mutated_val", val.Str())

	// 2. TargetSpanName
	traces := ptrace.NewTraces()
	span := traces.ResourceSpans().AppendEmpty().ScopeSpans().AppendEmpty().Spans().AppendEmpty()
	span.SetName("initial_span")
	tSpan := MutationTarget{Kind: TargetSpanName, Span: span}
	tSpan.Set("mutated_span")
	assert.Equal(t, "mutated_span", span.Name())

	// 3. TargetSpanEventName
	event := span.Events().AppendEmpty()
	event.SetName("initial_event")
	tEvent := MutationTarget{Kind: TargetSpanEventName, SpanEvent: event}
	tEvent.Set("mutated_event")
	assert.Equal(t, "mutated_event", event.Name())

	// 4. TargetMetricName & TargetMetricDescription
	metrics := pmetric.NewMetrics()
	metric := metrics.ResourceMetrics().AppendEmpty().ScopeMetrics().AppendEmpty().Metrics().AppendEmpty()
	metric.SetName("initial_metric_name")
	metric.SetDescription("initial_metric_desc")

	tMetricName := MutationTarget{Kind: TargetMetricName, Metric: metric}
	tMetricName.Set("mutated_metric_name")
	assert.Equal(t, "mutated_metric_name", metric.Name())

	tMetricDesc := MutationTarget{Kind: TargetMetricDescription, Metric: metric}
	tMetricDesc.Set("mutated_metric_desc")
	assert.Equal(t, "mutated_metric_desc", metric.Description())
}

func TestTable_DeduplicationAndMutation(t *testing.T) {
	table := NewTable(64)

	val1 := pcommon.NewValueStr("alice@example.com")
	val2 := pcommon.NewValueStr("alice@example.com")
	val3 := pcommon.NewValueStr("bob@example.com")

	idx1 := table.Add(val1.Str(), MutationTarget{Kind: TargetPCommonValue, Val: val1})
	idx2 := table.Add(val2.Str(), MutationTarget{Kind: TargetPCommonValue, Val: val2})
	idx3 := table.Add(val3.Str(), MutationTarget{Kind: TargetPCommonValue, Val: val3})

	assert.Equal(t, 0, idx1)
	assert.Equal(t, 0, idx2, "duplicate text should return existing index")
	assert.Equal(t, 1, idx3, "new text should get next index")

	uniques := table.UniqueStrings()
	require.Len(t, uniques, 2)
	assert.Equal(t, "alice@example.com", uniques[0])
	assert.Equal(t, "bob@example.com", uniques[1])

	targets0 := table.TargetsFor(0)
	require.Len(t, targets0, 2)
	for _, trg := range targets0 {
		trg.Set("[REDACTED_EMAIL]")
	}
	assert.Equal(t, "[REDACTED_EMAIL]", val1.Str())
	assert.Equal(t, "[REDACTED_EMAIL]", val2.Str())
	assert.Equal(t, "bob@example.com", val3.Str())
}

func TestTable_ResetAndCapacityRetention(t *testing.T) {
	table := NewTable(0) // tests default cap fallback

	for i := 0; i < 50; i++ {
		val := pcommon.NewValueStr("dup")
		table.Add(val.Str(), MutationTarget{Kind: TargetPCommonValue, Val: val})
	}
	assert.Len(t, table.UniqueStrings(), 1)

	// TargetsFor bounds check
	assert.Nil(t, table.TargetsFor(-1))
	assert.Nil(t, table.TargetsFor(100))

	// Reset clears without discarding map if below threshold
	table.Reset()
	assert.Empty(t, table.UniqueStrings())

	// Add multiple distinct items after reset to test slice capacity reuse
	for i := 0; i < 5; i++ {
		val := pcommon.NewValueStr("new_" + string(rune('a'+i)))
		idx := table.Add(val.Str(), MutationTarget{Kind: TargetPCommonValue, Val: val})
		assert.Equal(t, i, idx)
	}
	assert.Len(t, table.UniqueStrings(), 5)

	// Exceed maxRetainedCapacity to test memory bloat defense
	for i := 0; i < 2100; i++ {
		val := pcommon.NewValueStr("item_" + string(rune(i)))
		table.Add(val.Str(), MutationTarget{Kind: TargetPCommonValue, Val: val})
	}
	assert.Greater(t, len(table.UniqueStrings()), 2048)
	table.Reset()
	assert.Empty(t, table.UniqueStrings())
	assert.Empty(t, table.index)
}

func TestTable_AllocsOnDuplicates(t *testing.T) {
	table := NewTable(256)
	text := "repeated_log_template_string"
	val := pcommon.NewValueStr(text)
	target := MutationTarget{Kind: TargetPCommonValue, Val: val}

	// Warm-up table
	table.Add(text, target)
	for i := 0; i < 100; i++ {
		table.Add(text, target)
	}

	allocs := testing.AllocsPerRun(100, func() {
		table.Add(text, target)
	})
	assert.Equal(t, float64(0), allocs, "adding duplicate text should be 0 allocs/op")
}

func TestTable_AllocsAcrossReset(t *testing.T) {
	table := NewTable(64)
	val := pcommon.NewValueStr("test")
	target := MutationTarget{Kind: TargetPCommonValue, Val: val}

	strs := make([]string, 10)
	for i := 0; i < 10; i++ {
		strs[i] = string(rune('a' + i))
	}

	// Warm up 10 slots
	for i := 0; i < 10; i++ {
		table.Add(strs[i], target)
	}
	table.Reset()

	allocs := testing.AllocsPerRun(100, func() {
		for i := 0; i < 10; i++ {
			table.Add(strs[i], target)
		}
		table.Reset()
	})
	assert.Equal(t, float64(0), allocs, "adding known unique strings across Reset should be 0 allocs/op")
}
