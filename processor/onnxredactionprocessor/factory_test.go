package onnxredactionprocessor

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/consumer/consumertest"
	"go.opentelemetry.io/collector/processor/processortest"
)

func TestFactory_TypeAndDefaultConfig(t *testing.T) {
	fact := NewFactory()
	assert.Equal(t, Type, fact.Type())

	cfg := fact.CreateDefaultConfig()
	require.NotNil(t, cfg)
	pCfg, ok := cfg.(*Config)
	require.True(t, ok)
	assert.Equal(t, 64, pCfg.Inference.CoalesceMaxStrings)
	assert.Equal(t, 16384, pCfg.Inference.MaxPayloadBytes)
}

func TestFactory_CreateProcessorsMissingModel(t *testing.T) {
	fact := NewFactory()
	cfg := fact.CreateDefaultConfig()
	set := processortest.NewNopSettings(Type)

	// Logs
	lProc, err := fact.CreateLogs(context.Background(), set, cfg, consumertest.NewNop())
	assert.Error(t, err)
	assert.Nil(t, lProc)

	// Traces
	tProc, err := fact.CreateTraces(context.Background(), set, cfg, consumertest.NewNop())
	assert.Error(t, err)
	assert.Nil(t, tProc)

	// Metrics
	mProc, err := fact.CreateMetrics(context.Background(), set, cfg, consumertest.NewNop())
	assert.Error(t, err)
	assert.Nil(t, mProc)
}

func TestFactory_CreateProcessorsSuccess(t *testing.T) {
	fact := NewFactory()
	cfg := fact.CreateDefaultConfig().(*Config)
	cfg.Model.Path = "testdata/tiny_ner.onnx"
	cfg.Model.TokenizerPath = "testdata/tokenizer.json"
	cfg.Inference.WarmupOnStart = false
	set := processortest.NewNopSettings(Type)

	lProc, err := fact.CreateLogs(context.Background(), set, cfg, consumertest.NewNop())
	require.NoError(t, err)
	assert.NotNil(t, lProc)
	defer func() { _ = lProc.Shutdown(context.Background()) }()

	tProc, err := fact.CreateTraces(context.Background(), set, cfg, consumertest.NewNop())
	require.NoError(t, err)
	assert.NotNil(t, tProc)
	defer func() { _ = tProc.Shutdown(context.Background()) }()

	mProc, err := fact.CreateMetrics(context.Background(), set, cfg, consumertest.NewNop())
	require.NoError(t, err)
	assert.NotNil(t, mProc)
	_ = mProc.Shutdown(context.Background())
}

func TestFactory_SharedProcessorAcrossSignals(t *testing.T) {
	fact := NewFactory()
	cfg := fact.CreateDefaultConfig().(*Config)
	cfg.Model.Path = "testdata/tiny_ner.onnx"
	cfg.Model.TokenizerPath = "testdata/tokenizer.json"
	cfg.Inference.WarmupOnStart = false
	set := processortest.NewNopSettings(Type)

	lProc, err := fact.CreateLogs(context.Background(), set, cfg, consumertest.NewNop())
	require.NoError(t, err)
	defer func() { _ = lProc.Shutdown(context.Background()) }()

	tProc, err := fact.CreateTraces(context.Background(), set, cfg, consumertest.NewNop())
	require.NoError(t, err)
	defer func() { _ = tProc.Shutdown(context.Background()) }()

	mProc, err := fact.CreateMetrics(context.Background(), set, cfg, consumertest.NewNop())
	require.NoError(t, err)
	defer func() { _ = mProc.Shutdown(context.Background()) }()

	lp, ok := lProc.(*logsProcessor)
	require.True(t, ok)
	tp, ok := tProc.(*tracesProcessor)
	require.True(t, ok)
	mp, ok := mProc.(*metricsProcessor)
	require.True(t, ok)

	assert.Same(t, lp.inst.proc, tp.inst.proc, "logs and traces must share the same processor instance")
	assert.Same(t, lp.inst.proc, mp.inst.proc, "logs and metrics must share the same processor instance")
	assert.Equal(t, 3, lp.inst.refCount, "refCount must be 3 for the 3 signals")
}
