package onnxredactionprocessor

import (
	"context"
	"sync"

	"go.opentelemetry.io/collector/component"
	"go.opentelemetry.io/collector/consumer"
	"go.opentelemetry.io/collector/pdata/plog"
	"go.opentelemetry.io/collector/pdata/pmetric"
	"go.opentelemetry.io/collector/pdata/ptrace"
	"go.opentelemetry.io/collector/processor"
)

var (
	Type      = component.MustNewType("onnxredaction")
	stability = component.StabilityLevelAlpha
)

type sharedMap struct {
	mu         sync.Mutex
	processors map[*Config]*sharedInstance
}

type sharedInstance struct {
	proc      *redactionProcessor
	refCount  int
	startOnce sync.Once
	startErr  error
}

func newSharedMap() *sharedMap {
	return &sharedMap{
		processors: make(map[*Config]*sharedInstance),
	}
}

func (m *sharedMap) getOrCreate(cfg *Config, create func() (*redactionProcessor, error)) (*sharedInstance, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if inst, ok := m.processors[cfg]; ok {
		inst.refCount++
		return inst, nil
	}

	proc, err := create()
	if err != nil {
		return nil, err
	}

	inst := &sharedInstance{
		proc:     proc,
		refCount: 1,
	}
	m.processors[cfg] = inst
	return inst, nil
}

func (m *sharedMap) release(cfg *Config, ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	inst, ok := m.processors[cfg]
	if !ok {
		return nil
	}

	inst.refCount--
	if inst.refCount <= 0 {
		delete(m.processors, cfg)
		return inst.proc.Shutdown(ctx)
	}
	return nil
}

// NewFactory returns a new processor.Factory for the onnxredactionprocessor.
func NewFactory() processor.Factory {
	shared := newSharedMap()

	return processor.NewFactory(
		Type,
		createDefaultConfig,
		processor.WithLogs(createLogsProcessor(shared), stability),
		processor.WithTraces(createTracesProcessor(shared), stability),
		processor.WithMetrics(createMetricsProcessor(shared), stability),
	)
}

func createLogsProcessor(shared *sharedMap) processor.CreateLogsFunc {
	//nolint:gocritic // Implements OTel processor.CreateLogsFunc signature
	return func(
		_ context.Context,
		set processor.Settings,
		cfg component.Config,
		nextConsumer consumer.Logs,
	) (processor.Logs, error) {
		pCfg := cfg.(*Config)
		inst, err := shared.getOrCreate(pCfg, func() (*redactionProcessor, error) {
			return newProcessor(pCfg, &set.TelemetrySettings, nil, nil)
		})
		if err != nil {
			return nil, err
		}
		return &logsProcessor{
			shared: shared,
			cfg:    pCfg,
			inst:   inst,
			next:   nextConsumer,
		}, nil
	}
}

type logsProcessor struct {
	shared *sharedMap
	cfg    *Config
	inst   *sharedInstance
	next   consumer.Logs
}

func (l *logsProcessor) Start(ctx context.Context, host component.Host) error {
	l.inst.startOnce.Do(func() {
		l.inst.startErr = l.inst.proc.Start(ctx, host)
	})
	return l.inst.startErr
}

func (l *logsProcessor) Shutdown(ctx context.Context) error {
	return l.shared.release(l.cfg, ctx)
}

func (l *logsProcessor) ConsumeLogs(ctx context.Context, ld plog.Logs) error {
	if err := l.inst.proc.processLogs(ctx, ld); err != nil {
		return err
	}
	return l.next.ConsumeLogs(ctx, ld)
}

func (l *logsProcessor) Capabilities() consumer.Capabilities {
	return consumer.Capabilities{MutatesData: true}
}

func createTracesProcessor(shared *sharedMap) processor.CreateTracesFunc {
	//nolint:gocritic // Implements OTel processor.CreateTracesFunc signature
	return func(
		_ context.Context,
		set processor.Settings,
		cfg component.Config,
		nextConsumer consumer.Traces,
	) (processor.Traces, error) {
		pCfg := cfg.(*Config)
		inst, err := shared.getOrCreate(pCfg, func() (*redactionProcessor, error) {
			return newProcessor(pCfg, &set.TelemetrySettings, nil, nil)
		})
		if err != nil {
			return nil, err
		}
		return &tracesProcessor{
			shared: shared,
			cfg:    pCfg,
			inst:   inst,
			next:   nextConsumer,
		}, nil
	}
}

type tracesProcessor struct {
	shared *sharedMap
	cfg    *Config
	inst   *sharedInstance
	next   consumer.Traces
}

func (t *tracesProcessor) Start(ctx context.Context, host component.Host) error {
	t.inst.startOnce.Do(func() {
		t.inst.startErr = t.inst.proc.Start(ctx, host)
	})
	return t.inst.startErr
}

func (t *tracesProcessor) Shutdown(ctx context.Context) error {
	return t.shared.release(t.cfg, ctx)
}

func (t *tracesProcessor) ConsumeTraces(ctx context.Context, td ptrace.Traces) error {
	if err := t.inst.proc.processTraces(ctx, td); err != nil {
		return err
	}
	return t.next.ConsumeTraces(ctx, td)
}

func (t *tracesProcessor) Capabilities() consumer.Capabilities {
	return consumer.Capabilities{MutatesData: true}
}

func createMetricsProcessor(shared *sharedMap) processor.CreateMetricsFunc {
	//nolint:gocritic // Implements OTel processor.CreateMetricsFunc signature
	return func(
		_ context.Context,
		set processor.Settings,
		cfg component.Config,
		nextConsumer consumer.Metrics,
	) (processor.Metrics, error) {
		pCfg := cfg.(*Config)
		inst, err := shared.getOrCreate(pCfg, func() (*redactionProcessor, error) {
			return newProcessor(pCfg, &set.TelemetrySettings, nil, nil)
		})
		if err != nil {
			return nil, err
		}
		return &metricsProcessor{
			shared: shared,
			cfg:    pCfg,
			inst:   inst,
			next:   nextConsumer,
		}, nil
	}
}

type metricsProcessor struct {
	shared *sharedMap
	cfg    *Config
	inst   *sharedInstance
	next   consumer.Metrics
}

func (m *metricsProcessor) Start(ctx context.Context, host component.Host) error {
	m.inst.startOnce.Do(func() {
		m.inst.startErr = m.inst.proc.Start(ctx, host)
	})
	return m.inst.startErr
}

func (m *metricsProcessor) Shutdown(ctx context.Context) error {
	return m.shared.release(m.cfg, ctx)
}

func (m *metricsProcessor) ConsumeMetrics(ctx context.Context, md pmetric.Metrics) error {
	if err := m.inst.proc.processMetrics(ctx, md); err != nil {
		return err
	}
	return m.next.ConsumeMetrics(ctx, md)
}

func (m *metricsProcessor) Capabilities() consumer.Capabilities {
	return consumer.Capabilities{MutatesData: true}
}
