# syntax=docker/dockerfile:1
FROM golang:1.27-bookworm AS builder

ARG TARGETARCH
ARG ORT_VERSION=1.31.0
ARG OCB_VERSION=0.163.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    tar \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Download and verify ONNX Runtime and ocb based on TARGETARCH
RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) \
            ORT_ARCH="x64"; \
            ORT_SHA256="cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57"; \
            OCB_ARCH="linux_amd64"; \
            OCB_SHA256="2aea100793a07cdf396972ca6ab3b137163367399045a2052bd89ba92362a4d5"; \
            ;; \
        arm64) \
            ORT_ARCH="aarch64"; \
            ORT_SHA256="c5b8b3cca31f3d643a3b313b8f42e2d0f16a4fcefd02cda343d97bd5306afaab"; \
            OCB_ARCH="linux_arm64"; \
            OCB_SHA256="d8c9453a5eb95131441eca1678cb3e925d12e8a14848ceb19b7c0f41e973caf3"; \
            ;; \
        *) \
            echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; \
            exit 1; \
            ;; \
    esac; \
    curl -sSL -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz"; \
    echo "${ORT_SHA256}  onnxruntime.tgz" | sha256sum -c -; \
    mkdir -p /opt/onnxruntime; \
    tar -xzf onnxruntime.tgz -C /opt/onnxruntime --strip-components=1; \
    rm onnxruntime.tgz; \
    curl -sSL -o /usr/local/bin/ocb "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/cmd%2Fbuilder%2Fv${OCB_VERSION}/ocb_${OCB_VERSION}_${OCB_ARCH}"; \
    echo "${OCB_SHA256}  /usr/local/bin/ocb" | sha256sum -c -; \
    chmod +x /usr/local/bin/ocb

WORKDIR /src
COPY . .

ENV CGO_CFLAGS="-I/opt/onnxruntime/include"
ENV CGO_LDFLAGS="-L/opt/onnxruntime/lib -lonnxruntime -Wl,-rpath,/usr/local/lib"

# Build custom collector distribution
RUN ocb --config=builder-config.yaml

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy native ONNX Runtime libraries
COPY --from=builder /opt/onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

# Create non-root user and directories
RUN groupadd -g 10001 otel && useradd -u 10001 -g otel -s /bin/false otel \
    && mkdir -p /etc/otel/models /var/log/otel \
    && chown -R otel:otel /etc/otel /var/log/otel

# Copy collector binary, configs, and models
COPY --from=builder /src/bin/otelcol-onnxredactor /usr/local/bin/otelcol
COPY --from=builder /src/config.yaml /etc/otel/config.yaml
COPY --from=builder /src/processor/onnxredactionprocessor/testdata/tiny_ner.onnx /etc/otel/models/tiny_ner.onnx
COPY --from=builder /src/processor/onnxredactionprocessor/testdata/tokenizer.json /etc/otel/models/tokenizer.json

RUN chown -R otel:otel /etc/otel/models

USER 10001:10001

EXPOSE 4317 4318 8888

ENTRYPOINT ["/usr/local/bin/otelcol", "--config=/etc/otel/config.yaml"]
