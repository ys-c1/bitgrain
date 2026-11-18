# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Pluggable entropy-coding interface (`entropy_coder`): the encoder drives
  any backend through `init` / `encode_symbol` / `finish`; the coefficient
  walk is shared and monomorphized per plane class (zero per-symbol dispatch
  cost, no buffer copies across the interface).
- Zero-order adaptive arithmetic coding backend (`arith`), selectable with
  `BITGRAIN_ENTROPY=arith`; writes new stream versions 20 (RGB) / 21 (RGBA),
  decodable by `bitgrain_decode` like any other version.
- `BITGRAIN_ENTROPY` backend switch: `huffman` (default, byte-identical to
  the previous hardcoded path), `legacy` (pre-interface reference path),
  `arith`.
- Format spec: documented the v20/v21 arithmetic plane payload (FORMAT.md).
- Tests: byte-exact legacy-vs-interface equality, arithmetic round-trips,
  cross-backend pixel equality, and a 100 MiB benchmark
  (`cargo test --release -- --ignored --nocapture entropy_bench_100mb`).

### Changed
- Plane length prefixes are now reserved up front and patched in place,
  removing the per-plane buffer copy in the Huffman path.

## [2.0.0] - 2026-04-26

### Added
- Scheduler-friendly parallel execution with adaptive thresholds for encode/decode.
- Extended SIMD coverage (SSE2/AVX2/NEON) in colorspace/dequant hot paths.
- Installable shared libraries with ABI symlinks:
  - `libbitgrain.so.<ABI_MAJOR>`
  - `libbitgrain-simd.so.<ABI_MAJOR>`
- `pkg-config` metadata: `bitgrain.pc`.
- CMake consumer config: `BitgrainConfig.cmake`.
- Environment-based thread controls:
  - `BITGRAIN_THREADS`
  - `BITGRAIN_THREADS_CAP`

### Changed
- Improved library installation layout for external consumers.
- Expanded API contract docs for error-message lifetime and decode failure semantics.
- Updated Rust crate metadata repository URL.

### Notes
- The public C API remains in `includes/encoder.h`.
- Semantic versioning policy:
  - Patch/minor: source-compatible C API changes
  - Major: potential ABI break
