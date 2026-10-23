package bucket

import (
	"sort"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// Bucketer groups token windows into sub-batches according to sequence length thresholds
// and dynamic batch limits, with tight per-sub-batch sequence padding.
type Bucketer struct {
	limits             []int
	maxTensorBatchSize int
}

// New constructs a Bucketer with length bucket limits and max tensor batch size.
func New(limits []int, maxTensorBatchSize int) *Bucketer {
	if len(limits) == 0 {
		limits = []int{32, 128, 256}
	}
	sortedLimits := make([]int, len(limits))
	copy(sortedLimits, limits)
	sort.Ints(sortedLimits)

	if maxTensorBatchSize <= 0 {
		maxTensorBatchSize = 32
	}

	return &Bucketer{
		limits:             sortedLimits,
		maxTensorBatchSize: maxTensorBatchSize,
	}
}

// Group partitions windows into buckets, divides each bucket into sub-batches <= maxTensorBatchSize,
// and sets MaxSeqLen to the maximum window length in each sub-batch.
func (b *Bucketer) Group(windows []types.TokenWindow) []types.BucketedBatch {
	if len(windows) == 0 {
		return nil
	}

	// Partition windows into buckets
	bucketCount := len(b.limits) + 1
	buckets := make([][]types.TokenWindow, bucketCount)

	for _, w := range windows {
		seqLen := len(w.TokenIDs)
		bucketIdx := len(b.limits) // default to overflow bucket
		for i, lim := range b.limits {
			if seqLen <= lim {
				bucketIdx = i
				break
			}
		}
		buckets[bucketIdx] = append(buckets[bucketIdx], w)
	}

	var result []types.BucketedBatch

	for _, bWindows := range buckets {
		if len(bWindows) == 0 {
			continue
		}

		for start := 0; start < len(bWindows); start += b.maxTensorBatchSize {
			end := start + b.maxTensorBatchSize
			if end > len(bWindows) {
				end = len(bWindows)
			}

			sub := bWindows[start:end]
			actualMax := 0
			for _, w := range sub {
				if len(w.TokenIDs) > actualMax {
					actualMax = len(w.TokenIDs)
				}
			}

			result = append(result, types.BucketedBatch{
				MaxSeqLen: actualMax,
				Windows:   sub,
			})
		}
	}

	return result
}
