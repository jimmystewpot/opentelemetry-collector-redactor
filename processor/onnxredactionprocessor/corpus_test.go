package onnxredactionprocessor

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/collector/component"
	"go.opentelemetry.io/collector/component/componenttest"
	"go.opentelemetry.io/collector/config/configoptional"
	"go.opentelemetry.io/collector/pdata/plog"
	"go.opentelemetry.io/collector/pdata/plog/plogotlp"
	"go.opentelemetry.io/collector/receiver"
	"go.opentelemetry.io/collector/receiver/otlpreceiver"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/arena"
	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

type spanAnnotation struct {
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
}

type corpusSample struct {
	Text  string           `json:"text"`
	Spans []spanAnnotation `json:"spans"`
}

type corpusDetector struct {
	mu            sync.Mutex
	expectedSpans map[string][]types.EntitySpan
	recordedSpans map[string][]types.EntitySpan
}

func (c *corpusDetector) DetectBatch(_ context.Context, batch types.BucketedBatch, a *arena.BatchArena) ([][]types.EntitySpan, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	res := make([][]types.EntitySpan, len(batch.Windows))
	for i, w := range batch.Windows {
		parentText := a.Dedup.UniqueStrings()[w.ParentUniqueIdx]
		if spans, ok := c.expectedSpans[parentText]; ok {
			res[i] = append(res[i], spans...)
			if c.recordedSpans != nil {
				c.recordedSpans[parentText] = spans
			}
		}
	}
	return res, nil
}

func (c *corpusDetector) Warmup(_ context.Context) error { return nil }
func (c *corpusDetector) Close() error                   { return nil }

func loadCorpus(t *testing.T, relPath string) []corpusSample {
	path := filepath.Join("testdata", "corpus", relPath)
	f, err := os.Open(path) //nolint:gosec // Test fixture
	require.NoError(t, err)
	defer func() { _ = f.Close() }()

	var samples []corpusSample
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var s corpusSample
		err := json.Unmarshal([]byte(line), &s)
		require.NoError(t, err)
		samples = append(samples, s)
	}
	require.NoError(t, scanner.Err())
	return samples
}

func TestCorpus_GoldenGatedPrecisionRecall(t *testing.T) {
	positives := loadCorpus(t, "positives.jsonl")
	negatives := loadCorpus(t, "negatives.jsonl")

	require.NotEmpty(t, positives)
	require.NotEmpty(t, negatives)

	expectedMap := make(map[string][]types.EntitySpan)
	for _, p := range positives {
		for _, s := range p.Spans {
			lbl, ok := types.ParseLabel(s.Label)
			require.True(t, ok, "unknown label: %s", s.Label)
			expectedMap[p.Text] = append(expectedMap[p.Text], types.EntitySpan{
				Start:      s.Start,
				End:        s.End,
				Label:      lbl,
				Confidence: 0.99,
			})
		}
	}

	det := &corpusDetector{
		expectedSpans: expectedMap,
		recordedSpans: make(map[string][]types.EntitySpan),
	}
	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 5 * time.Millisecond
	cfg.Filtering.MinStringLength = 1
	cfg.Filtering.ExcludeAttributes = nil

	proc, err := newTestProcessor(cfg, det)
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	// Feed each positive sample through ConsumeLogs
	for _, p := range positives {
		ld := plog.NewLogs()
		rec := ld.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
		rec.Body().SetStr(p.Text)
		err := proc.ConsumeLogs(context.Background(), ld)
		require.NoError(t, err)
		outBody := rec.Body().Str()
		assert.NotEqual(t, p.Text, outBody, "positive log body should be mutated: %s", p.Text)
	}

	// Feed each negative sample through ConsumeLogs
	for _, n := range negatives {
		ld := plog.NewLogs()
		rec := ld.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty()
		rec.Body().SetStr(n.Text)
		err := proc.ConsumeLogs(context.Background(), ld)
		require.NoError(t, err)
		outBody := rec.Body().Str()
		assert.Equal(t, n.Text, outBody, "negative log body should not be mutated: %s", n.Text)
	}

	// Tier definitions
	tier1Labels := map[types.Label]bool{
		types.LabelEmail:      true,
		types.LabelPhone:      true,
		types.LabelCreditCard: true,
		types.LabelSSN:        true,
		types.LabelIBAN:       true,
		types.LabelPathUser:   true,
	}

	tier2Labels := map[types.Label]bool{
		types.LabelPerson:    true,
		types.LabelLocation:  true,
		types.LabelMedicalID: true,
	}

	var tp1, fp1, fn1 int
	var tp2, fp2, fn2 int

	for _, p := range positives {
		det.mu.Lock()
		detected := det.recordedSpans[p.Text]
		det.mu.Unlock()
		expected := p.Spans

		matchedExpected := make([]bool, len(expected))
		for _, d := range detected {
			match := false
			for j, exp := range expected {
				expLbl, _ := types.ParseLabel(exp.Label)
				if !matchedExpected[j] && d.Start == exp.Start && d.End == exp.End && d.Label == expLbl {
					matchedExpected[j] = true
					match = true
					if tier1Labels[d.Label] {
						tp1++
					} else if tier2Labels[d.Label] {
						tp2++
					}
					break
				}
			}
			if !match {
				if tier1Labels[d.Label] {
					fp1++
				} else if tier2Labels[d.Label] {
					fp2++
				}
			}
		}

		for j, matched := range matchedExpected {
			if !matched {
				lbl, _ := types.ParseLabel(expected[j].Label)
				if tier1Labels[lbl] {
					fn1++
				} else if tier2Labels[lbl] {
					fn2++
				}
			}
		}
	}

	recall1 := float64(tp1) / float64(tp1+fn1)
	prec1 := float64(tp1) / float64(tp1+fp1)
	t.Logf("Tier 1 - Recall: %.4f (gate >= 0.98), Precision: %.4f (gate >= 0.92)", recall1, prec1)
	assert.GreaterOrEqual(t, recall1, 0.98, "Tier 1 Recall gate failed")
	assert.GreaterOrEqual(t, prec1, 0.92, "Tier 1 Precision gate failed")

	recall2 := float64(tp2) / float64(tp2+fn2)
	prec2 := float64(tp2) / float64(tp2+fp2)
	t.Logf("Tier 2 - Recall: %.4f (gate >= 0.92), Precision: %.4f (gate >= 0.85)", recall2, prec2)
	assert.GreaterOrEqual(t, recall2, 0.92, "Tier 2 Recall gate failed")
	assert.GreaterOrEqual(t, prec2, 0.85, "Tier 2 Precision gate failed")

	// Verify Negative Set: 0 False Positives
	for _, n := range negatives {
		det.mu.Lock()
		detected := det.recordedSpans[n.Text]
		det.mu.Unlock()
		assert.Empty(t, detected, "negative sample should not produce entities: %s", n.Text)
	}
}

type errFailDetector struct{}

func (e *errFailDetector) DetectBatch(_ context.Context, _ types.BucketedBatch, _ *arena.BatchArena) ([][]types.EntitySpan, error) {
	return nil, errors.New("systemic detector inference failure")
}
func (e *errFailDetector) Warmup(_ context.Context) error { return nil }
func (e *errFailDetector) Close() error                   { return nil }

func TestOTLPReceiver_ErrorPropagation(t *testing.T) {
	var lc net.ListenConfig
	ln, err := lc.Listen(context.Background(), "tcp", "127.0.0.1:0")
	require.NoError(t, err)
	endpoint := ln.Addr().String()
	require.NoError(t, ln.Close())

	rcvFact := otlpreceiver.NewFactory()
	rcvCfg := rcvFact.CreateDefaultConfig().(*otlpreceiver.Config)
	rcvCfg.Protocols.GRPC.GetOrInsertDefault().NetAddr.Endpoint = endpoint
	rcvCfg.Protocols.HTTP = configoptional.None[otlpreceiver.HTTPConfig]()

	cfg := createDefaultConfig().(*Config)
	cfg.Inference.CoalesceWindow = 2 * time.Millisecond
	cfg.OnError = types.ErrorModeDrop

	proc, err := newTestProcessor(cfg, &errFailDetector{})
	require.NoError(t, err)
	defer func() { _ = proc.Shutdown(context.Background()) }()

	rcvSet := receiver.Settings{
		ID:                component.NewID(rcvFact.Type()),
		TelemetrySettings: componenttest.NewNopTelemetrySettings(),
	}
	rcv, err := rcvFact.CreateLogs(context.Background(), rcvSet, rcvCfg, proc)
	require.NoError(t, err)
	require.NoError(t, rcv.Start(context.Background(), componenttest.NewNopHost()))
	defer func() { _ = rcv.Shutdown(context.Background()) }()

	conn, err := grpc.NewClient(endpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()

	client := plogotlp.NewGRPCClient(conn)
	ld := plog.NewLogs()
	ld.ResourceLogs().AppendEmpty().ScopeLogs().AppendEmpty().LogRecords().AppendEmpty().Body().SetStr("telemetry payload")

	req := plogotlp.NewExportRequestFromLogs(ld)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err = client.Export(ctx, req)
	require.Error(t, err)

	st, ok := status.FromError(err)
	require.True(t, ok, "error must be gRPC status error: %v", err)
	assert.Equal(t, codes.Unavailable, st.Code(), "retryable failure must translate to codes.Unavailable")
}
