GO ?= go
GOPATH ?= $(shell $(GO) env GOPATH)
GOLANGCI_LINT ?= $(shell which golangci-lint 2>/dev/null || echo $(GOPATH)/bin/golangci-lint)

ORT_VERSION ?= 1.31.0
ORT_DIR ?= $(CURDIR)/.ort

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    ORT_OS = osx
    ifeq ($(UNAME_M),arm64)
        ORT_ARCH = arm64
    else
        ORT_ARCH = x86_64
    endif
    LIB_EXT = dylib
else
    ORT_OS = linux
    ifeq ($(UNAME_M),aarch64)
        ORT_ARCH = aarch64
    else ifeq ($(UNAME_M),arm64)
        ORT_ARCH = aarch64
    else
        ORT_ARCH = x64
    endif
    LIB_EXT = so
endif

ORT_ARCHIVE = onnxruntime-$(ORT_OS)-$(ORT_ARCH)-$(ORT_VERSION).tgz
ORT_URL = https://github.com/microsoft/onnxruntime/releases/download/v$(ORT_VERSION)/$(ORT_ARCHIVE)

CGO_CFLAGS ?= -I$(ORT_DIR)/include
CGO_LDFLAGS ?= -L$(ORT_DIR)/lib -lonnxruntime -Wl,-rpath,$(ORT_DIR)/lib

export CGO_CFLAGS
export CGO_LDFLAGS

.PHONY: all fmt lint test bench clean download-ort

all: fmt lint test

fmt:
	@cd processor/onnxredactionprocessor && $(GO) fmt ./...

lint:
	@cd processor/onnxredactionprocessor && $(GOLANGCI_LINT) run ./...

test:
	@cd processor/onnxredactionprocessor && $(GO) test -race -count=1 ./...

bench:
	@cd processor/onnxredactionprocessor && $(GO) test -bench=. -benchmem -run=^$$ ./...

download-ort:
	@if [ ! -d "$(ORT_DIR)/include" ]; then \
		echo "Downloading ONNX Runtime $(ORT_VERSION) for $(ORT_OS)-$(ORT_ARCH)..."; \
		mkdir -p $(ORT_DIR); \
		curl -sSL -o /tmp/$(ORT_ARCHIVE) $(ORT_URL); \
		tar -xzf /tmp/$(ORT_ARCHIVE) -C /tmp; \
		cp -r /tmp/onnxruntime-$(ORT_OS)-$(ORT_ARCH)-$(ORT_VERSION)/* $(ORT_DIR)/; \
		rm -rf /tmp/$(ORT_ARCHIVE) /tmp/onnxruntime-$(ORT_OS)-$(ORT_ARCH)-$(ORT_VERSION); \
		echo "ONNX Runtime unpacked into $(ORT_DIR)"; \
	else \
		echo "ONNX Runtime already present at $(ORT_DIR)"; \
	fi

clean:
	@rm -rf bin/ $(ORT_DIR)
