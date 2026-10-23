package onnxredactionprocessor

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/consumer/consumertest"
	"go.opentelemetry.io/collector/pdata/plog"
	"go.opentelemetry.io/collector/processor/processortest"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/mask"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

func BenchmarkMask_CleanString(b *testing.B) {
	cleanText := "Standard application log message without any sensitive personal data"
	scratch := make([]byte, 512)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := mask.Mask(cleanText, nil, 16384, types.ErrorModeDrop, scratch)
		if err != nil || out == "" {
			b.Fatal("unexpected mask error")
		}
	}
}

func BenchmarkMask_DirtyString(b *testing.B) {
	dirtyText := "Contact alice@example.com or admin at C:\\Users\\Administrator\\secret.log"
	spans := []types.EntitySpan{
		{Start: 8, End: 25, Label: types.LabelEmail, Confidence: 0.99},
		{Start: 47, End: 60, Label: types.LabelPathUser, Confidence: 0.99},
	}
	scratch := make([]byte, 512)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := mask.Mask(dirtyText, spans, 16384, types.ErrorModeDrop, scratch)
		if err != nil || out == "" {
			b.Fatal("unexpected mask error")
		}
	}
}

func BenchmarkPipeline_FullInference(b *testing.B) {
	fact := NewFactory()
	cfg := fact.CreateDefaultConfig().(*Config)
	cfg.Model.Path = "testdata/tiny_ner.onnx"
	cfg.Model.TokenizerPath = "testdata/tokenizer.json"
	cfg.Inference.CoalesceWindow = 1 * time.Millisecond
	cfg.Inference.CoalesceMaxStrings = 32
	cfg.Inference.WarmupOnStart = true

	set := processortest.NewNopSettings(Type)
	proc, err := fact.CreateLogs(context.Background(), set, cfg, consumertest.NewNop())
	require.NoError(b, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	ld := plog.NewLogs()
	rl := ld.ResourceLogs().AppendEmpty()
	rl.Resource().Attributes().PutStr("service.name", "billing")
	rec := rl.ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
	rec.Body().SetStr("Customer invoice sent to alice@example.com for service")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := proc.ConsumeLogs(context.Background(), ld); err != nil {
			b.Fatal(err)
		}
	}
}
