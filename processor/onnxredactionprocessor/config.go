package onnxredactionprocessor

import (
	"errors"
	"fmt"
	"time"

	"go.opentelemetry.io/collector/component"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// SessionConfig configures execution provider threading and runtime options for ONNX Runtime.
type SessionConfig struct {
	IntraOpThreads int  `mapstructure:"intra_op_threads"`
	ConcurrentRuns int  `mapstructure:"concurrent_runs"`
	AllowSpinning  bool `mapstructure:"allow_spinning"`
}

// ModelConfig configures model and tokenizer files, integrity hashes, and session settings.
type ModelConfig struct {
	Path            string        `mapstructure:"path"`
	SHA256          string        `mapstructure:"sha256"`
	TokenizerPath   string        `mapstructure:"tokenizer_path"`
	TokenizerSHA256 string        `mapstructure:"tokenizer_sha256"`
	Session         SessionConfig `mapstructure:"session"`
}

// InferenceConfig configures the batched sliding-window inference pipeline.
type InferenceConfig struct {
	CoalesceWindow      time.Duration `mapstructure:"coalesce_window"`
	CoalesceMaxStrings  int           `mapstructure:"coalesce_max_strings"`
	MaxTensorBatchSize  int           `mapstructure:"max_tensor_batch_size"`
	LengthBuckets       []int         `mapstructure:"length_buckets"`
	WindowTokens        int           `mapstructure:"window_tokens"`
	WindowOverlapTokens int           `mapstructure:"window_overlap_tokens"`
	MaxPayloadBytes     int           `mapstructure:"max_payload_bytes"`
	ConfidenceThreshold float64       `mapstructure:"confidence_threshold"`
	Timeout             time.Duration `mapstructure:"timeout"`
	WarmupOnStart       bool          `mapstructure:"warmup_on_start"`
}

// FilteringConfig configures pre-inference static attribute bypass and string length thresholds.
type FilteringConfig struct {
	MinStringLength   int      `mapstructure:"min_string_length"`
	ExcludeAttributes []string `mapstructure:"exclude_attributes"`
}

// Config defines the configuration for the onnxredactionprocessor.
type Config struct {
	Model     ModelConfig     `mapstructure:"model"`
	Inference InferenceConfig `mapstructure:"inference"`
	Entities  []string        `mapstructure:"entities"`
	Filtering FilteringConfig `mapstructure:"filtering"`
	OnError   types.ErrorMode `mapstructure:"on_error"`
	DryRun    bool            `mapstructure:"dry_run"`
}

var _ component.Config = (*Config)(nil)

func createDefaultConfig() component.Config {
	return &Config{
		Model: ModelConfig{
			Path:            "",
			SHA256:          "",
			TokenizerPath:   "",
			TokenizerSHA256: "",
			Session: SessionConfig{
				IntraOpThreads: 2,
				ConcurrentRuns: 2,
				AllowSpinning:  false,
			},
		},
		Inference: InferenceConfig{
			CoalesceWindow:      2 * time.Millisecond,
			CoalesceMaxStrings:  64,
			MaxTensorBatchSize:  32,
			LengthBuckets:       []int{32, 128, 256},
			WindowTokens:        256,
			WindowOverlapTokens: 32,
			MaxPayloadBytes:     16384,
			ConfidenceThreshold: 0.85,
			Timeout:             25 * time.Millisecond,
			WarmupOnStart:       true,
		},
		Entities: []string{
			"PERSON", "EMAIL", "PHONE", "LOCATION", "CREDIT_CARD", "SSN", "IBAN", "MEDICAL_ID", "PATH_USER",
		},
		Filtering: FilteringConfig{
			MinStringLength: 4,
			ExcludeAttributes: []string{
				"service.name", "service.version", "host.*", "k8s.*", "container.*", "telemetry.sdk.*", "os.*",
			},
		},
		OnError: types.ErrorModeDrop,
		DryRun:  false,
	}
}

// Validate checks the configuration against all operational constraints.
func (c *Config) Validate() error {
	var errs []error

	if c.Model.Path == "" {
		errs = append(errs, errors.New("model.path must not be empty"))
	}
	if c.Model.TokenizerPath == "" {
		errs = append(errs, errors.New("model.tokenizer_path must not be empty"))
	}
	if c.Model.Session.IntraOpThreads <= 0 {
		errs = append(errs, errors.New("model.session.intra_op_threads must be > 0"))
	}
	if c.Model.Session.ConcurrentRuns <= 0 {
		errs = append(errs, errors.New("model.session.concurrent_runs must be > 0"))
	}

	if c.Inference.CoalesceWindow <= 0 {
		errs = append(errs, errors.New("inference.coalesce_window must be > 0"))
	}
	if c.Inference.CoalesceMaxStrings <= 0 {
		errs = append(errs, errors.New("inference.coalesce_max_strings must be > 0"))
	}
	if c.Inference.MaxTensorBatchSize <= 0 {
		errs = append(errs, errors.New("inference.max_tensor_batch_size must be > 0"))
	}
	if len(c.Inference.LengthBuckets) == 0 {
		errs = append(errs, errors.New("inference.length_buckets must not be empty"))
	} else {
		prev := 0
		for i, b := range c.Inference.LengthBuckets {
			if b <= prev {
				errs = append(errs, fmt.Errorf("inference.length_buckets[%d]=%d must be strictly greater than previous bucket %d", i, b, prev))
				break
			}
			prev = b
		}
		if prev > c.Inference.WindowTokens {
			errs = append(errs, fmt.Errorf("largest length bucket (%d) exceeds inference.window_tokens (%d)", prev, c.Inference.WindowTokens))
		}
	}

	if c.Inference.WindowTokens <= 0 {
		errs = append(errs, errors.New("inference.window_tokens must be > 0"))
	}
	if c.Inference.WindowOverlapTokens < 0 || c.Inference.WindowOverlapTokens >= c.Inference.WindowTokens {
		errs = append(errs, errors.New("inference.window_overlap_tokens must be >= 0 and < inference.window_tokens"))
	}
	if c.Inference.MaxPayloadBytes <= 0 {
		errs = append(errs, errors.New("inference.max_payload_bytes must be > 0"))
	}
	if c.Inference.ConfidenceThreshold <= 0.0 || c.Inference.ConfidenceThreshold > 1.0 {
		errs = append(errs, errors.New("inference.confidence_threshold must be between 0.0 (exclusive) and 1.0 (inclusive)"))
	}
	if c.Inference.Timeout <= 0 {
		errs = append(errs, errors.New("inference.timeout must be > 0"))
	}

	if len(c.Entities) == 0 {
		errs = append(errs, errors.New("entities list must not be empty"))
	}
	for _, ent := range c.Entities {
		if lbl, ok := types.ParseLabel(ent); !ok || lbl == types.LabelNone {
			errs = append(errs, fmt.Errorf("unrecognized entity label: %q", ent))
		}
	}

	if c.Filtering.MinStringLength < 0 {
		errs = append(errs, errors.New("filtering.min_string_length must be >= 0"))
	}

	switch c.OnError {
	case types.ErrorModeDrop, types.ErrorModeIgnore:
		// Valid
	default:
		errs = append(errs, fmt.Errorf("invalid on_error mode %q: must be either %q or %q", c.OnError, types.ErrorModeDrop, types.ErrorModeIgnore))
	}

	return errors.Join(errs...)
}
