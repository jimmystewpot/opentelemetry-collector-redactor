package filter

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAttributeFilter_ShouldInspect(t *testing.T) {
	patterns := []string{
		"service.name",
		"service.version",
		"host.*",
		"k8s.*",
		"container.*",
		"telemetry.sdk.*",
		"os.*",
	}

	f := New(4, patterns)

	tests := []struct {
		name     string
		key      string
		val      string
		expected bool
	}{
		// Excluded attributes (exact)
		{"exact match service.name", "service.name", "billing-service", false},
		{"exact match service.version", "service.version", "v1.2.3", false},

		// Excluded attributes (prefix glob)
		{"prefix glob host.name", "host.name", "ip-10-0-1-42", false},
		{"prefix glob k8s.pod.name", "k8s.pod.name", "my-pod-xyz", false},
		{"prefix glob container.id", "container.id", "docker-12345", false},
		{"prefix glob telemetry.sdk.language", "telemetry.sdk.language", "go", false},
		{"prefix glob os.type", "os.type", "linux", false},

		// Short string filtering (< minStringLen = 4)
		{"empty value", "custom.attr", "", false},
		{"single char", "custom.attr", "a", false},
		{"three chars", "custom.attr", "foo", false},
		{"three runes multi-byte", "custom.attr", "日a本", false}, // 3 runes, even if 7 bytes

		// Pure numeric strings
		{"pure integer", "custom.attr", "12345", false},
		{"pure negative float", "custom.attr", "-42.50", false},
		{"pure positive float", "custom.attr", "+3.14159", false},
		{"zero value", "custom.attr", "0000", false},
		{"single plus sign", "custom.attr", "+", false},
		{"single minus sign", "custom.attr", "-", false},

		// Positive cases (must be inspected)
		{"ip address with multiple dots", "custom.attr", "1.2.3.4", true},
		{"two dots numeric", "custom.attr", "12.34.56", true},
		{"http url", "http.url", "https://example.com/user/alice", true},
		{"db statement", "db.statement", "SELECT * FROM users WHERE email='alice@example.com'", true},
		{"file path", "log.file.path", "/home/alice/.ssh/id_rsa", true},
		{"message with name", "log.message", "John Doe authenticated successfully", true},
		{"four char word", "custom.attr", "word", true},
		{"alphanumeric with digits", "custom.attr", "user12345", true},
		{"credit card with dashes", "card.number", "4532-1234-5678-9010", true},
		{"negative number with text", "status", "-42 error code", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := f.ShouldInspect(tt.key, tt.val)
			assert.Equal(t, tt.expected, got, "key=%q, val=%q", tt.key, tt.val)
		})
	}
}

func TestAttributeFilter_EmptyConfig(t *testing.T) {
	f := New(0, []string{"", "prefix*"})
	assert.False(t, f.ShouldInspect("key", "")) // empty string still skipped
	assert.True(t, f.ShouldInspect("key", "hello"))
	assert.False(t, f.ShouldInspect("prefix.sub", "hello"))
}

func TestIsPureNumeric_Direct(t *testing.T) {
	assert.False(t, isPureNumeric(""))
	assert.False(t, isPureNumeric("+"))
	assert.False(t, isPureNumeric("-"))
	assert.False(t, isPureNumeric("12.34.56"))
	assert.True(t, isPureNumeric("123"))
	assert.True(t, isPureNumeric("-12.34"))
}
