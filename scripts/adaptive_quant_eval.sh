#!/usr/bin/env bash
#
# One-shot offline runner for the perceptual adaptive-quantization feature.
#
# It runs the repository's existing Rust test suite (which now includes the
# adaptive-quantization unit and evaluation tests) with cargo forced into
# offline mode, then re-prints the metrics/curve-scan report. No third-party
# crate, network access, or build-system change is required: everything
# builds from the sources already vendored in this repository.
#
# Usage:  scripts/adaptive_quant_eval.sh
# Exit:   0 only if every cargo test passes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$ROOT_DIR/rust"

# Refuse to reach the network: all dependencies are cached/vendored locally.
export CARGO_NET_OFFLINE=true
export CARGO_TARGET_DIR="$RUST_DIR/target"

echo "==> Running full Rust test suite offline (cargo test --release)"
cd "$RUST_DIR"
cargo test --release

echo
echo "==> Adaptive-quantization metrics + mapping-curve scan (re-printed)"
# --nocapture surfaces the println! tables; the test is deterministic and
# reuses the in-memory 32x32 four-region synthetic sample.
cargo test --release -- adaptive_quant_metrics_and_curve_scan --nocapture
