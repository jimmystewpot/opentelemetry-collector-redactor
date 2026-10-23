package filter

import (
	"strings"
	"unicode/utf8"
)

// AttributeFilter evaluates telemetry attribute keys and string values
// to skip unnecessary inspection before entering the tokenization and inference pipeline.
type AttributeFilter struct {
	minStringLen   int
	exactExcludes  map[string]struct{}
	prefixExcludes []string
}

// New constructs an AttributeFilter from minStringLen and wildcard/exact exclusion patterns.
func New(minStringLen int, excludePatterns []string) *AttributeFilter {
	exact := make(map[string]struct{})
	var prefixes []string

	for _, p := range excludePatterns {
		p = strings.TrimSpace(p)
		switch {
		case strings.HasSuffix(p, ".*"), strings.HasSuffix(p, "*"):
			prefixes = append(prefixes, strings.TrimSuffix(p, "*"))
		case p != "":
			exact[p] = struct{}{}
		}
	}

	return &AttributeFilter{
		minStringLen:   minStringLen,
		exactExcludes:  exact,
		prefixExcludes: prefixes,
	}
}

// ShouldInspect returns true if the key/val attribute pair should undergo NER inspection.
// Returns false if:
// 1. Value is empty or shorter than minStringLen (by byte or rune count).
// 2. Key matches an exact exclusion pattern.
// 3. Key matches a prefix exclusion pattern.
// 4. Value is purely numeric (integer or float literal).
func (f *AttributeFilter) ShouldInspect(key, val string) bool {
	if val == "" {
		return false
	}
	if len(val) < f.minStringLen {
		return false
	}
	if f.minStringLen > 0 && utf8.RuneCountInString(val) < f.minStringLen {
		return false
	}

	if len(f.exactExcludes) > 0 {
		if _, ok := f.exactExcludes[key]; ok {
			return false
		}
	}

	for _, p := range f.prefixExcludes {
		if strings.HasPrefix(key, p) {
			return false
		}
	}

	return !isPureNumeric(val)
}

func isPureNumeric(s string) bool {
	if s == "" {
		return false
	}
	start := 0
	if s[0] == '+' || s[0] == '-' {
		if len(s) == 1 {
			return false
		}
		start = 1
	}

	hasDigit := false
	hasDot := false

	for i := start; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= '0' && c <= '9':
			hasDigit = true
		case c == '.':
			if hasDot {
				return false
			}
			hasDot = true
		default:
			return false
		}
	}

	return hasDigit
}
