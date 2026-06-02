FROM debian:bookworm-slim

# ---------------------------------------------------------------------------
# cross-lang-verifier — hermetic external replication image (Step 54).
#
# A stranger runs:
#
#     docker build -t cross-lang-verifier .
#     docker run --rm cross-lang-verifier        # == make reproduce-kit
#
# and regenerates every byte-reproducible table and re-confirms every oracle
# against the *real* toolchain (clang/UBSan, rustc, go, z3/boolector). The
# toolchain versions are pinned so the artifact is reproducible.
# ---------------------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/usr/local/go/bin:/opt/venv/bin:$PATH

# Base build/runtime tools, clang+UBSan, z3 and boolector (the solver portfolio),
# Python, plus curl/ca-certs for the rustup/go installers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        llvm \
        z3 \
        boolector \
        python3 \
        python3-venv \
        python3-pip \
        curl \
        ca-certificates \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

# Rust (pinned) — provides rustc/cargo and the MIR ownership facts the ingester
# relies on.
ARG RUST_VERSION=1.91.1
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y \
        --profile minimal --default-toolchain ${RUST_VERSION} \
    && rustc --version && cargo --version

# Go (pinned) — the second target language pair.
ARG GO_VERSION=1.25.5
RUN ARCH="$(dpkg --print-architecture)" \
    && case "$ARCH" in amd64) GOARCH=amd64;; arm64) GOARCH=arm64;; *) GOARCH="$ARCH";; esac \
    && curl -sSfL "https://go.dev/dl/go${GO_VERSION}.linux-${GOARCH}.tar.gz" -o /tmp/go.tgz \
    && tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz \
    && go version

# A dedicated venv with z3 (the in-process solver) and pytest.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir z3-solver pytest

WORKDIR /work
COPY . /work

# The image's default action is the full replication kit.
ENV PYTHON=/opt/venv/bin/python
CMD ["make", "reproduce-kit"]
