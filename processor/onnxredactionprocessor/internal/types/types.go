package types

import (
	"strings"

	"go.opentelemetry.io/collector/pdata/pcommon"
)

// Label represents a supported Named Entity Recognition (NER) category.
type Label uint8

const (
	LabelNone Label = iota
	LabelPerson
	LabelEmail
	LabelPhone
	LabelLocation
	LabelCreditCard
	LabelSSN
	LabelIBAN
	LabelMedicalID
	LabelPathUser
)

// String returns the canonical uppercase label name.
func (l Label) String() string {
	switch l {
	case LabelPerson:
		return "PERSON"
	case LabelEmail:
		return "EMAIL"
	case LabelPhone:
		return "PHONE"
	case LabelLocation:
		return "LOCATION"
	case LabelCreditCard:
		return "CREDIT_CARD"
	case LabelSSN:
		return "SSN"
	case LabelIBAN:
		return "IBAN"
	case LabelMedicalID:
		return "MEDICAL_ID"
	case LabelPathUser:
		return "PATH_USER"
	default:
		return "NONE"
	}
}

// Placeholder returns the formatted mask string, e.g. "[REDACTED_PERSON]".
func (l Label) Placeholder() string {
	if l == LabelNone {
		return ""
	}
	return "[REDACTED_" + l.String() + "]"
}

// ParseLabel parses a case-insensitive string into a Label enum.
func ParseLabel(s string) (Label, bool) {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "PERSON":
		return LabelPerson, true
	case "EMAIL":
		return LabelEmail, true
	case "PHONE":
		return LabelPhone, true
	case "LOCATION":
		return LabelLocation, true
	case "CREDIT_CARD":
		return LabelCreditCard, true
	case "SSN":
		return LabelSSN, true
	case "IBAN":
		return LabelIBAN, true
	case "MEDICAL_ID":
		return LabelMedicalID, true
	case "PATH_USER":
		return LabelPathUser, true
	default:
		return LabelNone, false
	}
}

// RedactedUninspected is the uniform replacement marker for uninspectable strings or oversize tails.
const RedactedUninspected = "[REDACTED_UNINSPECTED]"

// EntitySpan represents a half-open byte range [Start, End) in a UTF-8 string with associated label and confidence score.
type EntitySpan struct {
	Start      int
	End        int
	Label      Label
	Confidence float32
}

// Offset maps a single token to its half-open byte range [Start, End) in the parent string.
type Offset struct {
	Start int
	End   int
}

// TokenWindow is a slice of up to WindowTokens token IDs from a single parent string.
type TokenWindow struct {
	ParentUniqueIdx int
	TokenIDs        []int64 // int64 unified: zero conversion pass to ONNX tensors
	Offsets         []Offset
}

// BucketedBatch groups TokenWindows by their padded sequence length.
type BucketedBatch struct {
	MaxSeqLen int
	Windows   []TokenWindow
}

// SignalKind identifies the OpenTelemetry telemetry signal type.
type SignalKind uint8

const (
	SignalLogs SignalKind = iota
	SignalTraces
	SignalMetrics
)

// ErrorMode controls failure policy handling (fail closed vs fail open).
type ErrorMode string

const (
	ErrorModeDrop   ErrorMode = "drop"
	ErrorModeIgnore ErrorMode = "ignore"
)

// TargetKind classifies the source and role of a telemetry target string.
type TargetKind uint8

const (
	TargetKindLogBody TargetKind = iota
	TargetKindLogAttr
	TargetKindSpanName
	TargetKindSpanAttr
	TargetKindSpanEventName
	TargetKindSpanEventAttr
	TargetKindMetricName
	TargetKindMetricDescription
	TargetKindMetricAttr
)

// MutationTarget represents an in-place telemetry target to mutate with redacted text.
// Designed as a value struct (zero closures) to eliminate heap allocations on the hot path.
type MutationTarget struct {
	Value pcommon.Value
	Kind  TargetKind
}
