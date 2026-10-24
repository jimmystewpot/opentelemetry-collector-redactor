package tokenizer

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/jimmystewpot/vector-onnx-redactor/processor/onnxredactionprocessor/internal/types"
)

// Tokenizer converts raw UTF-8 text into int64 token IDs and byte-exact half-open string offsets.
type Tokenizer interface {
	Encode(text string, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset, error)
	Close() error
}

type hfTokenizerJSON struct {
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Model struct {
		Type                    string         `json:"type"`
		UnkToken                string         `json:"unk_token"`
		ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
		MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
		Vocab                   map[string]int `json:"vocab"`
	} `json:"model"`
}

type wordPieceTokenizer struct {
	vocab        map[string]int64
	unkID        int64
	clsID        int64
	sepID        int64
	prefix       string
	maxInputChar int
}

// NewWordPieceTokenizer loads a HuggingFace tokenizer.json file, verifies its SHA-256 checksum,
// and initializes an in-process, allocation-minimized WordPiece tokenizer.
func NewWordPieceTokenizer(tokenizerJSONPath, expectedSHA256 string) (Tokenizer, error) {
	cleanPath := filepath.Clean(tokenizerJSONPath)
	data, err := os.ReadFile(cleanPath) //nolint:gosec // File path is configured via collector config
	if err != nil {
		return nil, fmt.Errorf("tokenizer: failed to read file %s: %w", cleanPath, err)
	}

	if expectedSHA256 != "" {
		h := sha256.Sum256(data)
		actualSHA := hex.EncodeToString(h[:])
		if !strings.EqualFold(actualSHA, expectedSHA256) {
			return nil, fmt.Errorf("tokenizer: checksum mismatch for %s: expected %s, got %s",
				tokenizerJSONPath, expectedSHA256, actualSHA)
		}
	}

	var parsed hfTokenizerJSON
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, fmt.Errorf("tokenizer: failed to parse tokenizer JSON %s: %w", tokenizerJSONPath, err)
	}

	vocab := make(map[string]int64, len(parsed.Model.Vocab))
	for k, v := range parsed.Model.Vocab {
		vocab[k] = int64(v)
	}

	unkID := int64(100)
	if v, ok := vocab[parsed.Model.UnkToken]; ok {
		unkID = v
	}

	clsID := int64(101)
	if v, ok := vocab["[CLS]"]; ok {
		clsID = v
	}

	sepID := int64(102)
	if v, ok := vocab["[SEP]"]; ok {
		sepID = v
	}

	for _, at := range parsed.AddedTokens {
		switch at.Content {
		case "[CLS]":
			clsID = int64(at.ID)
		case "[SEP]":
			sepID = int64(at.ID)
		case "[UNK]":
			unkID = int64(at.ID)
		}
	}

	prefix := parsed.Model.ContinuingSubwordPrefix
	if prefix == "" {
		prefix = "##"
	}

	maxChars := parsed.Model.MaxInputCharsPerWord
	if maxChars <= 0 {
		maxChars = 100
	}

	return &wordPieceTokenizer{
		vocab:        vocab,
		unkID:        unkID,
		clsID:        clsID,
		sepID:        sepID,
		prefix:       prefix,
		maxInputChar: maxChars,
	}, nil
}

func (t *wordPieceTokenizer) Close() error {
	return nil
}

// Encode converts text into token IDs and string byte offsets.
func (t *wordPieceTokenizer) Encode(text string, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset, error) {
	if !utf8.ValidString(text) {
		return nil, nil, fmt.Errorf("tokenizer: text is not valid UTF-8")
	}

	textLen := len(text)
	// Add [CLS]
	dstIDs = append(dstIDs, t.clsID)
	dstOffsets = append(dstOffsets, types.Offset{Start: 0, End: 0})

	if textLen > 0 {
		// Tokenize words and punctuation
		byteIdx := 0
		for byteIdx < textLen {
			r, size := utf8.DecodeRuneInString(text[byteIdx:])
			if unicode.IsSpace(r) {
				byteIdx += size
				continue
			}

			// Punctuation, symbols, or CJK character treated as individual single-rune tokens
			if isPunctOrCJK(r) {
				var runeBuf [utf8.UTFMax]byte
				n := utf8.EncodeRune(runeBuf[:], unicode.ToLower(r))
				id, ok := t.vocab[string(runeBuf[:n])]
				if !ok {
					id = t.unkID
				}
				dstIDs = append(dstIDs, id)
				dstOffsets = append(dstOffsets, types.Offset{Start: byteIdx, End: byteIdx + size})
				byteIdx += size
				continue
			}

			// Word: sequence of non-space, non-punct, non-CJK runes
			wordStart := byteIdx
			for byteIdx < textLen {
				rNext, sizeNext := utf8.DecodeRuneInString(text[byteIdx:])
				if unicode.IsSpace(rNext) || isPunctOrCJK(rNext) {
					break
				}
				byteIdx += sizeNext
			}
			wordEnd := byteIdx

			// Apply WordPiece subword tokenization
			dstIDs, dstOffsets = t.tokenizeWord(text, wordStart, wordEnd, dstIDs, dstOffsets)
		}
	}

	// Add [SEP]
	dstIDs = append(dstIDs, t.sepID)
	dstOffsets = append(dstOffsets, types.Offset{Start: textLen, End: textLen})

	return dstIDs, dstOffsets, nil
}

func (t *wordPieceTokenizer) tokenizeWord(text string, wordStart, wordEnd int, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset) {
	rawLen := wordEnd - wordStart
	if rawLen > t.maxInputChar {
		dstIDs = append(dstIDs, t.unkID)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart, End: wordEnd})
		return dstIDs, dstOffsets
	}

	prefixLen := len(t.prefix)
	if prefixLen+rawLen > 256 {
		return t.tokenizeWordSlow(text, wordStart, wordEnd, dstIDs, dstOffsets)
	}

	var buf [256]byte

	// Fast path: ASCII lowercasing inline
	isASCII := true
	for i := 0; i < rawLen; i++ {
		b := text[wordStart+i]
		if b >= utf8.RuneSelf {
			isASCII = false
			break
		}
		if b >= 'A' && b <= 'Z' {
			b += 'a' - 'A'
		}
		buf[prefixLen+i] = b
	}

	lowercasedLen := rawLen
	hasRuneLenChange := false
	var textOffsets [257]uint8

	if !isASCII {
		srcRel := 0
		dstRel := 0
		for srcRel < rawLen {
			textOffsets[dstRel] = uint8(srcRel)
			r, sz := utf8.DecodeRuneInString(text[wordStart+srcRel : wordEnd])
			rLow := unicode.ToLower(r)
			encSz := utf8.EncodeRune(buf[prefixLen+dstRel:], rLow)
			if encSz != sz {
				hasRuneLenChange = true
			}
			srcRel += sz
			dstRel += encSz
			if prefixLen+dstRel > len(buf) {
				return t.tokenizeWordSlow(text, wordStart, wordEnd, dstIDs, dstOffsets)
			}
		}
		textOffsets[dstRel] = uint8(srcRel)
		lowercasedLen = dstRel
	}

	if lowercasedLen > t.maxInputChar {
		dstIDs = append(dstIDs, t.unkID)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart, End: wordEnd})
		return dstIDs, dstOffsets
	}

	// First try exact whole word match
	if id, ok := t.vocab[string(buf[prefixLen:prefixLen+lowercasedLen])]; ok {
		dstIDs = append(dstIDs, id)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart, End: wordEnd})
		return dstIDs, dstOffsets
	}

	// Subword segmentation
	subStart := 0
	isBad := false
	startTokenCount := len(dstIDs)

	var savedPrefix [32]byte
	canUseSavedPrefix := prefixLen <= len(savedPrefix)

	for subStart < lowercasedLen {
		subEnd := lowercasedLen
		curID := int64(-1)

		if subStart > 0 {
			if !canUseSavedPrefix {
				return t.tokenizeWordSlow(text, wordStart, wordEnd, dstIDs[:startTokenCount], dstOffsets[:startTokenCount])
			}
			copy(savedPrefix[:prefixLen], buf[subStart:prefixLen+subStart])
			copy(buf[subStart:prefixLen+subStart], t.prefix)
		}

		for subStart < subEnd {
			var subStr string
			if subStart == 0 {
				subStr = string(buf[prefixLen : prefixLen+subEnd])
			} else {
				subStr = string(buf[subStart : prefixLen+subEnd])
			}

			if id, ok := t.vocab[subStr]; ok {
				curID = id
				break
			}
			// Backtrack by rune from buf[prefixLen+subStart : prefixLen+subEnd]
			_, lastRuneSize := utf8.DecodeLastRune(buf[prefixLen+subStart : prefixLen+subEnd])
			subEnd -= lastRuneSize
		}

		if subStart > 0 {
			copy(buf[subStart:prefixLen+subStart], savedPrefix[:prefixLen])
		}

		if curID == -1 {
			isBad = true
			break
		}

		startOff := wordStart + subStart
		endOff := wordStart + subEnd
		if hasRuneLenChange {
			startOff = wordStart + int(textOffsets[subStart])
			endOff = wordStart + int(textOffsets[subEnd])
		}

		dstIDs = append(dstIDs, curID)
		dstOffsets = append(dstOffsets, types.Offset{Start: startOff, End: endOff})
		subStart = subEnd
	}

	if isBad {
		// Roll back any partial subwords and emit single UNK token
		dstIDs = dstIDs[:startTokenCount]
		dstOffsets = dstOffsets[:startTokenCount]
		dstIDs = append(dstIDs, t.unkID)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart, End: wordEnd})
	}

	return dstIDs, dstOffsets
}

func (t *wordPieceTokenizer) tokenizeWordSlow(text string, wordStart, wordEnd int, dstIDs []int64, dstOffsets []types.Offset) ([]int64, []types.Offset) {
	word := strings.ToLower(text[wordStart:wordEnd])
	if len(word) > t.maxInputChar {
		dstIDs = append(dstIDs, t.unkID)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart, End: wordEnd})
		return dstIDs, dstOffsets
	}

	// First try exact whole word match
	if id, ok := t.vocab[word]; ok {
		dstIDs = append(dstIDs, id)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart, End: wordEnd})
		return dstIDs, dstOffsets
	}

	// Subword segmentation
	subStart := 0
	wordLen := len(word)
	isBad := false

	startTokenCount := len(dstIDs)

	for subStart < wordLen {
		subEnd := wordLen
		curID := int64(-1)

		for subStart < subEnd {
			subStr := word[subStart:subEnd]
			if subStart > 0 {
				subStr = t.prefix + subStr
			}

			if id, ok := t.vocab[subStr]; ok {
				curID = id
				break
			}
			// Backtrack by rune
			_, lastRuneSize := utf8.DecodeLastRuneInString(word[subStart:subEnd])
			subEnd -= lastRuneSize
		}

		if curID == -1 {
			isBad = true
			break
		}

		dstIDs = append(dstIDs, curID)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart + subStart, End: wordStart + subEnd})
		subStart = subEnd
	}

	if isBad {
		// Roll back any partial subwords and emit single UNK token
		dstIDs = dstIDs[:startTokenCount]
		dstOffsets = dstOffsets[:startTokenCount]
		dstIDs = append(dstIDs, t.unkID)
		dstOffsets = append(dstOffsets, types.Offset{Start: wordStart, End: wordEnd})
	}

	return dstIDs, dstOffsets
}

func isPunctOrCJK(r rune) bool {
	if r == '_' || r == '-' {
		return false
	}
	if unicode.IsPunct(r) || unicode.IsSymbol(r) {
		return true
	}
	// CJK Unified Ideographs & Hangul/Hiragana/Katakana
	if (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x3040 && r <= 0x309F) || // Hiragana
		(r >= 0x30A0 && r <= 0x30FF) || // Katakana
		(r >= 0xAC00 && r <= 0xD7AF) { // Hangul Syllables
		return true
	}
	return false
}
