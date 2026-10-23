package onnxredactionprocessor

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/confmap/confmaptest"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func validBaselineConfig() *Config {
	return &Config{
		Model: ModelConfig{
			Path:            "/etc/otel/models/pii_ner_int8.onnx",
			SHA256:          "7ef06dbfe51de3cd4cb08c2075a390b21ad261b28e7c7f9e36e6e66d2d7329a8",
			TokenizerPath:   "/etc/otel/models/tokenizer.json",
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

func TestLoadConfig(t *testing.T) {
	cm, err := confmaptest.LoadConf(filepath.Join("testdata", "config_valid.yaml"))
	require.NoError(t, err)

	cfg := createDefaultConfig().(*Config)
	err = cm.Unmarshal(cfg)
	require.NoError(t, err)

	assert.Equal(t, "/etc/otel/models/pii_ner_int8.onnx", cfg.Model.Path)
	assert.Equal(t, "7ef06dbfe51de3cd4cb08c2075a390b21ad261b28e7c7f9e36e6e66d2d7329a8", cfg.Model.SHA256)
	assert.Equal(t, "/etc/otel/models/tokenizer.json", cfg.Model.TokenizerPath)
	assert.Equal(t, 2, cfg.Model.Session.IntraOpThreads)
	assert.Equal(t, 2, cfg.Model.Session.ConcurrentRuns)
	assert.False(t, cfg.Model.Session.AllowSpinning)

	assert.Equal(t, 2*time.Millisecond, cfg.Inference.CoalesceWindow)
	assert.Equal(t, 64, cfg.Inference.CoalesceMaxStrings)
	assert.Equal(t, 32, cfg.Inference.MaxTensorBatchSize)
	assert.Equal(t, []int{32, 128, 256}, cfg.Inference.LengthBuckets)
	assert.Equal(t, 256, cfg.Inference.WindowTokens)
	assert.Equal(t, 32, cfg.Inference.WindowOverlapTokens)
	assert.Equal(t, 16384, cfg.Inference.MaxPayloadBytes)
	assert.InDelta(t, 0.85, cfg.Inference.ConfidenceThreshold, 0.001)
	assert.Equal(t, 25*time.Millisecond, cfg.Inference.Timeout)
	assert.True(t, cfg.Inference.WarmupOnStart)

	assert.Equal(t, types.ErrorModeDrop, cfg.OnError)
	assert.False(t, cfg.DryRun)

	err = cfg.Validate()
	require.NoError(t, err)
}

func TestConfigValidate(t *testing.T) {
	tests := []struct {
		name        string
		mutate      func(*Config)
		errContains string
	}{
		{
			name: "missing model path",
			mutate: func(c *Config) {
				c.Model.Path = ""
			},
			errContains: "model.path",
		},
		{
			name: "missing tokenizer path",
			mutate: func(c *Config) {
				c.Model.TokenizerPath = ""
			},
			errContains: "model.tokenizer_path",
		},
		{
			name: "invalid session intra_op_threads",
			mutate: func(c *Config) {
				c.Model.Session.IntraOpThreads = 0
			},
			errContains: "intra_op_threads",
		},
		{
			name: "invalid session concurrent_runs",
			mutate: func(c *Config) {
				c.Model.Session.ConcurrentRuns = 0
			},
			errContains: "concurrent_runs",
		},
		{
			name: "invalid coalesce_window",
			mutate: func(c *Config) {
				c.Inference.CoalesceWindow = 0
			},
			errContains: "coalesce_window",
		},
		{
			name: "invalid coalesce_max_strings",
			mutate: func(c *Config) {
				c.Inference.CoalesceMaxStrings = 0
			},
			errContains: "coalesce_max_strings",
		},
		{
			name: "invalid max_tensor_batch_size",
			mutate: func(c *Config) {
				c.Inference.MaxTensorBatchSize = 0
			},
			errContains: "max_tensor_batch_size",
		},
		{
			name: "empty length_buckets",
			mutate: func(c *Config) {
				c.Inference.LengthBuckets = nil
			},
			errContains: "length_buckets",
		},
		{
			name: "unsorted length_buckets",
			mutate: func(c *Config) {
				c.Inference.LengthBuckets = []int{128, 32}
			},
			errContains: "length_buckets",
		},
		{
			name: "length_bucket exceeds window_tokens",
			mutate: func(c *Config) {
				c.Inference.LengthBuckets = []int{32, 512}
			},
			errContains: "window_tokens",
		},
		{
			name: "invalid window_tokens",
			mutate: func(c *Config) {
				c.Inference.WindowTokens = 0
			},
			errContains: "window_tokens",
		},
		{
			name: "window_overlap >= window_tokens",
			mutate: func(c *Config) {
				c.Inference.WindowOverlapTokens = 256
			},
			errContains: "window_overlap_tokens",
		},
		{
			name: "invalid max_payload_bytes",
			mutate: func(c *Config) {
				c.Inference.MaxPayloadBytes = 0
			},
			errContains: "max_payload_bytes",
		},
		{
			name: "invalid confidence_threshold <= 0",
			mutate: func(c *Config) {
				c.Inference.ConfidenceThreshold = 0.0
			},
			errContains: "confidence_threshold",
		},
		{
			name: "invalid confidence_threshold > 1",
			mutate: func(c *Config) {
				c.Inference.ConfidenceThreshold = 1.05
			},
			errContains: "confidence_threshold",
		},
		{
			name: "invalid timeout",
			mutate: func(c *Config) {
				c.Inference.Timeout = 0
			},
			errContains: "timeout",
		},
		{
			name: "empty entities",
			mutate: func(c *Config) {
				c.Entities = nil
			},
			errContains: "entities",
		},
		{
			name: "unknown entity",
			mutate: func(c *Config) {
				c.Entities = []string{"PERSON", "NON_EXISTENT_ENTITY"}
			},
			errContains: "NON_EXISTENT_ENTITY",
		},
		{
			name: "negative min_string_length",
			mutate: func(c *Config) {
				c.Filtering.MinStringLength = -1
			},
			errContains: "min_string_length",
		},
		{
			name: "invalid on_error",
			mutate: func(c *Config) {
				c.OnError = "invalid_mode"
			},
			errContains: "on_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := validBaselineConfig()
			tt.mutate(cfg)
			err := cfg.Validate()
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.errContains)
		})
	}
}
