package dedup

import (
	"go.opentelemetry.io/collector/pdata/pcommon"
	"go.opentelemetry.io/collector/pdata/pmetric"
	"go.opentelemetry.io/collector/pdata/ptrace"
)

// TargetKind identifies which telemetry field a MutationTarget modifies.
type TargetKind uint8

const (
	TargetPCommonValue      TargetKind = iota // Mutates pcommon.Value.SetStr
	TargetSpanName                            // Mutates ptrace.Span.SetName
	TargetSpanEventName                       // Mutates ptrace.SpanEvent.SetName
	TargetMetricName                          // Mutates pmetric.Metric.SetName
	TargetMetricDescription                   // Mutates pmetric.Metric.SetDescription
	TargetSpanStatusMessage                   // Mutates ptrace.Span.Status().SetMessage
)

// MutationTarget is a value-type descriptor that applies mutations in-place with zero closures.
type MutationTarget struct {
	Kind       TargetKind
	Val        pcommon.Value
	Span       ptrace.Span
	SpanEvent  ptrace.SpanEvent
	Metric     pmetric.Metric
	MetricName string
	Signal     string
}

// Set mutates the underlying telemetry field in-place.
func (t *MutationTarget) Set(s string) {
	switch t.Kind {
	case TargetPCommonValue:
		t.Val.SetStr(s)
	case TargetSpanName:
		t.Span.SetName(s)
	case TargetSpanEventName:
		t.SpanEvent.SetName(s)
	case TargetMetricName:
		t.Metric.SetName(s)
	case TargetMetricDescription:
		t.Metric.SetDescription(s)
	case TargetSpanStatusMessage:
		t.Span.Status().SetMessage(s)
	}
}

const maxRetainedCapacity = 2048

// Table maintains a deduplication index of unique strings and their associated mutation targets.
type Table struct {
	initialCap    int
	index         map[string]int
	uniqueStrings []string
	targets       [][]MutationTarget
}

// NewTable initializes a deduplication Table with the specified initial capacity.
func NewTable(initialCap int) *Table {
	if initialCap <= 0 {
		initialCap = 64
	}
	return &Table{
		initialCap:    initialCap,
		index:         make(map[string]int, initialCap),
		uniqueStrings: make([]string, 0, initialCap),
		targets:       make([][]MutationTarget, 0, initialCap),
	}
}

// Add inserts or finds a text string and associates the target with its unique index.
//
//nolint:gocritic // MutationTarget is a value-type descriptor avoiding heap-allocated closures
func (t *Table) Add(text string, target MutationTarget) int {
	if idx, ok := t.index[text]; ok {
		t.targets[idx] = append(t.targets[idx], target)
		return idx
	}

	idx := len(t.uniqueStrings)
	t.index[text] = idx
	t.uniqueStrings = append(t.uniqueStrings, text)

	if idx < cap(t.targets) {
		t.targets = t.targets[:idx+1]
		if cap(t.targets[idx]) > 0 {
			t.targets[idx] = append(t.targets[idx][:0], target)
		} else {
			t.targets[idx] = append(make([]MutationTarget, 0, 4), target)
		}
	} else {
		targetSlice := make([]MutationTarget, 0, 4)
		targetSlice = append(targetSlice, target)
		t.targets = append(t.targets, targetSlice)
	}

	return idx
}

// UniqueStrings returns the ordered slice of distinct strings collected in the batch.
func (t *Table) UniqueStrings() []string {
	return t.uniqueStrings
}

// TargetsFor returns the slice of targets referencing the given unique string index.
func (t *Table) TargetsFor(uniqueIdx int) []MutationTarget {
	if uniqueIdx < 0 || uniqueIdx >= len(t.uniqueStrings) {
		return nil
	}
	return t.targets[uniqueIdx]
}

// Reset clears the table for reuse. If the table grew beyond maxRetainedCapacity (2048),
// reallocates the map and slices to prevent permanent memory and bucket bloat.
func (t *Table) Reset() {
	if len(t.uniqueStrings) > maxRetainedCapacity || len(t.index) > maxRetainedCapacity {
		t.index = make(map[string]int, t.initialCap)
		t.uniqueStrings = make([]string, 0, t.initialCap)
		t.targets = make([][]MutationTarget, 0, t.initialCap)
		return
	}

	clear(t.index)
	t.uniqueStrings = t.uniqueStrings[:0]
	for i := range t.targets {
		t.targets[i] = t.targets[i][:0]
	}
	t.targets = t.targets[:0]
}
