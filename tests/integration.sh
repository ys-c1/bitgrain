#!/bin/bash
# Integration tests for bitgrain CLI. Run from repo root.
set -e
cd "$(dirname "$0")/.."
BIN="./bitgrain"

# Build if needed
if [ ! -x "$BIN" ]; then
    make bitgrain 2>/dev/null || { echo "Build failed (need libwebp)"; exit 1; }
fi

# Create minimal 8x8 grayscale PGM
mkdir -p tests/out
printf 'P5\n8 8\n255\n' > tests/out/mini.pgm
for i in {1..64}; do printf '\x80'; done >> tests/out/mini.pgm

# NOTE: the CLI loads every input as RGB when possible (see c/encode_cli.c,
# "Prefer true RGB"), so mini.pgm encodes to a v18 YCbCr stream and decodes
# to 3 channels. The .pgm writer only accepts 1-channel streams, so the
# decode/roundtrip outputs below use .png (RGB-capable). Decoding this
# fixture to .pgm fails by design on 2.0.0.

echo "=== Encode ==="
$BIN -i tests/out/mini.pgm -o tests/out/mini.bg -y
test -f tests/out/mini.bg || { echo "Encode failed"; exit 1; }

echo "=== Decode ==="
$BIN -d -i tests/out/mini.bg -o tests/out/mini_decoded.png -y
test -f tests/out/mini_decoded.png || { echo "Decode failed"; exit 1; }

echo "=== Round-trip ==="
$BIN -cd -i tests/out/mini.pgm -o tests/out/mini_rt.png -y -m
test -f tests/out/mini_rt.png || { echo "Round-trip failed"; exit 1; }

echo "=== All integration tests passed ==="
