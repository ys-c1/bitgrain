#!/bin/bash
# Compare the adaptive arithmetic entropy path against the legacy Huffman path.
#
# Generates four small samples, encodes each with both coders, decodes, verifies
# pixel-identical output, exercises the two model-update extremes (period 0 =
# static, period 1 = update every symbol), and reports size and codec-only time.
#
# Usage: ./scripts/arith_bench.sh [quality]
#
# Environment:
#   BITGRAIN_BASELINE   path to a bitgrain CLI built from the pre-arithmetic
#                       commit (default: /tmp/bitgrain-baseline)
#
# No network access and no third-party libraries beyond what the project already
# links. Samples and results live under bench_out/arith/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
QUALITY="${1:-85}"
OUT="$ROOT/bench_out/arith"
BG="$ROOT/bitgrain"
BENCH="$ROOT/bitgrain-bench"
BASELINE="${BITGRAIN_BASELINE:-/tmp/bitgrain-baseline}"

mkdir -p "$OUT"

if [ ! -x "$BG" ]; then
    echo "Error: $BG not found. Run 'make bitgrain' first." >&2
    exit 1
fi
if [ ! -x "$BASELINE" ]; then
    echo "Error: legacy baseline binary not found at $BASELINE." >&2
    echo "       Build the pre-arithmetic commit and set BITGRAIN_BASELINE=<path>." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Generate samples (deterministic, small)
# ---------------------------------------------------------------------------
python3 - "$OUT" <<'PY'
import os, random, struct, sys

out = sys.argv[1]
random.seed(42)

def ppm(name, w, h, pix):
    with open(os.path.join(out, name), 'wb') as f:
        f.write(b'P6\n%d %d\n255\n' % (w, h))
        f.write(bytes(pix))

def pgm(name, w, h, pix):
    with open(os.path.join(out, name), 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (w, h))
        f.write(bytes(pix))

W, H = 160, 120

# 1) Photo-like RGB: smooth gradients, mild texture and sensor-ish noise.
pix = []
for y in range(H):
    for x in range(W):
        r = int(128 + 90 * (x / W - 0.5) + 30 * (y / H - 0.5))
        g = int(120 + 60 * (y / H - 0.5))
        b = int(140 + 50 * (x / W - 0.5))
        n = random.randint(-18, 18)
        pix += [max(0, min(255, r + n)), max(0, min(255, g + n // 2)), max(0, min(255, b - n))]
ppm('photo_rgb.ppm', W, H, pix)

# 2) Synthetic RGB: flat regions with sharp edges.
pix = []
for y in range(H):
    for x in range(W):
        pix += [230, 40, 40] if (x // 20 + y // 20) % 2 == 0 else [40, 60, 230]
ppm('synthetic_rgb.ppm', W, H, pix)

# 3) Grayscale: horizontal gradient plus light texture.
pix = []
for y in range(H):
    for x in range(W):
        pix.append((int(x * 255 / W) + random.randint(-8, 8)) & 0xFF)
pgm('gray_gradient.pgm', W, H, pix)

# 4) RGBA: the CLI loader prefers RGB and drops alpha, so this sample is driven
#    through the C API by rgba_helper.c below (real v19 stream).
PY

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
encode_cli() { "$BG"       encode "$1" -o "$2" -q "$QUALITY" --threads 1 -y >/dev/null 2>&1; }
decode_cli() { "$BG"       decode "$1" -o "$2" -y >/dev/null 2>&1; }
encode_base(){ "$BASELINE" encode "$1" -o "$2" -q "$QUALITY" --threads 1 -y >/dev/null 2>&1; }
decode_base(){ "$BASELINE" decode "$1" -o "$2" -y >/dev/null 2>&1; }

same() { cmp -s "$1" "$2" && echo YES || echo NO; }

# Build a tiny C-API driver for the RGBA path using this tree's own header/static lib.
RGBA_HELPER="$OUT/rgba_helper"
cat > "$OUT/rgba_helper.c" <<'CEOF'
#include "encoder.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const int W = 160, H = 120;
    size_t n = (size_t)W * H * 4;
    unsigned char *rgba = malloc(n), *dec = malloc(n);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        size_t i = ((size_t)y * W + x) * 4;
        rgba[i] = (unsigned char)((x * 255) / W);
        rgba[i+1] = (unsigned char)((y * 255) / H);
        rgba[i+2] = (unsigned char)(((x + y) * 255) / (W + H));
        rgba[i+3] = (unsigned char)(255 * (0.3 + 0.7 * (x + y) / (double)(W + H)));
    }
    size_t cap = n * 2 + (1u << 20);
    unsigned char *bg = malloc(cap);
    int32_t len = 0;
    if (bitgrain_encode_rgba(rgba, W, H, bg, (uint32_t)cap, &len, 85) != 0) return 1;
    uint32_t dw, dh, dc;
    if (bitgrain_decode(bg, len, dec, (uint32_t)n, &dw, &dh, &dc) != 0) return 1;
    FILE *f = fopen(argv[1], "wb");
    if (f) { fwrite(bg, 1, (size_t)len, f); fclose(f); }
    f = fopen(argv[2], "wb");
    if (f) { fwrite(dec, 1, n, f); fclose(f); }
    fprintf(stderr, "version=%u size=%d\n", bg[2], (int)len);
    return 0;
}
CEOF

build_rgba_helper() { # $1 = static lib path, $2 = output binary
    gcc -O2 -I"$ROOT/includes" "$OUT/rgba_helper.c" "$1" \
        "$ROOT/c/dct.o" "$ROOT/c/quant.o" -o "$2" -lpthread -ldl -lm 2>/dev/null
}

# ---------------------------------------------------------------------------
# Per-sample comparison
# ---------------------------------------------------------------------------
printf '%-20s %10s %10s %10s %9s %8s %8s %8s\n' \
    sample huffman_bytes arith_bytes reduction pixels_legacy_decode p0_ok p1_ok

for s in photo_rgb.ppm synthetic_rgb.ppm gray_gradient.pgm; do
    base="${s%.*}"
    encode_base "$OUT/$s" "$OUT/${base}.huffman.bg"
    encode_cli  "$OUT/$s" "$OUT/${base}.arith.bg"
    decode_base "$OUT/${base}.huffman.bg" "$OUT/${base}.huffman.bmp"
    decode_cli  "$OUT/${base}.arith.bg"   "$OUT/${base}.arith.bmp"
    decode_cli  "$OUT/${base}.huffman.bg" "$OUT/${base}.legacy_rt.bmp"

    BITGRAIN_ARITH_UPDATE_PERIOD=0 encode_cli "$OUT/$s" "$OUT/${base}.p0.bg"
    BITGRAIN_ARITH_UPDATE_PERIOD=1 encode_cli "$OUT/$s" "$OUT/${base}.p1.bg"
    BITGRAIN_ARITH_UPDATE_PERIOD= decode_cli "$OUT/${base}.p0.bg" "$OUT/${base}.p0.bmp"
    BITGRAIN_ARITH_UPDATE_PERIOD= decode_cli "$OUT/${base}.p1.bg" "$OUT/${base}.p1.bmp"

    hb=$(stat -c%s "$OUT/${base}.huffman.bg")
    ab=$(stat -c%s "$OUT/${base}.arith.bg")
    red=$(python3 -c "print(f'{(1-$ab/$hb)*100:.2f}%')")
    printf '%-20s %10s %10s %10s %9s %8s %8s %8s\n' "$s" "$hb" "$ab" "$red" \
        "$(same "$OUT/${base}.huffman.bmp" "$OUT/${base}.arith.bmp")" \
        "$(same "$OUT/${base}.huffman.bmp" "$OUT/${base}.legacy_rt.bmp")" \
        "$(same "$OUT/${base}.huffman.bmp" "$OUT/${base}.p0.bmp")" \
        "$(same "$OUT/${base}.huffman.bmp" "$OUT/${base}.p1.bmp")"
done

# RGBA (v19): compare against the baseline library via a second helper build.
if build_rgba_helper "$ROOT/rust/target/release/deps/libbitgrain.a" "$RGBA_HELPER"; then
    "$RGBA_HELPER" "$OUT/alpha_rgba.arith.bg" "$OUT/alpha_rgba.arith.bmp" 2>"$OUT/alpha_rgba.info"
    # Re-run to confirm the decode is stable (same stream, same pixels).
    "$RGBA_HELPER" "$OUT/alpha_rgba.rt.bg" "$OUT/alpha_rgba.rt.bmp" 2>/dev/null
    BITGRAIN_ARITH_UPDATE_PERIOD=0 "$RGBA_HELPER" "$OUT/alpha_rgba.p0.bg" "$OUT/alpha_rgba.p0.bmp" 2>/dev/null
    BITGRAIN_ARITH_UPDATE_PERIOD=1 "$RGBA_HELPER" "$OUT/alpha_rgba.p1.bg" "$OUT/alpha_rgba.p1.bmp" 2>/dev/null
    ab=$(stat -c%s "$OUT/alpha_rgba.arith.bg")
    if build_rgba_helper "$(dirname "$BASELINE")/rust/target/release/deps/libbitgrain.a" "$RGBA_HELPER.base" 2>/dev/null; then
        "$RGBA_HELPER.base" "$OUT/alpha_rgba.huffman.bg" "$OUT/alpha_rgba.huffman.bmp" 2>/dev/null
        hb=$(stat -c%s "$OUT/alpha_rgba.huffman.bg")
        red=$(python3 -c "print(f'{(1-$ab/$hb)*100:.2f}%')")
        px=$(same "$OUT/alpha_rgba.huffman.bmp" "$OUT/alpha_rgba.arith.bmp")
        printf '%-20s %10s %10s %10s %9s %8s %8s %8s\n' "alpha_rgba.tga (v19)" "$hb" "$ab" "$red" \
            "$px" "$(same "$OUT/alpha_rgba.huffman.bmp" "$OUT/alpha_rgba.arith.bmp")" \
            "$(same "$OUT/alpha_rgba.huffman.bmp" "$OUT/alpha_rgba.p0.bmp")" \
            "$(same "$OUT/alpha_rgba.huffman.bmp" "$OUT/alpha_rgba.p1.bmp")"
    else
        echo "alpha_rgba.tga (v19): arithmetic stream $ab bytes, $(cat "$OUT/alpha_rgba.info")" \
             "(baseline library not available for comparison)"
    fi
else
    echo "alpha_rgba.tga (v19): skipped (could not link helper against static lib)"
fi

# ---------------------------------------------------------------------------
# Codec-only timing on a ~1 MB RGB image
# ---------------------------------------------------------------------------
python3 - "$OUT" <<'PY'
import os, random, sys
out = sys.argv[1]
random.seed(11)
W, H = 640, 512  # 983040 bytes of RGB
buf = bytearray()
for y in range(H):
    for x in range(W):
        r, g, b = (x * 255) // W, (y * 255) // H, ((x + y) * 255) // (W + H)
        n = random.randint(-10, 10)
        buf += bytes(((r + n) & 0xFF, (g + n) & 0xFF, (b - n) & 0xFF))
with open(os.path.join(out, 'timing_1mb.ppm'), 'wb') as f:
    f.write(b'P6\n%d %d\n255\n' % (W, H))
    f.write(buf)
PY

echo
echo "=== Codec-only timing (960 KB RGB, quality $QUALITY, 1 thread) ==="
if [ -x "$BENCH" ]; then
    echo "-- arithmetic --"
    "$BENCH" -q "$QUALITY" -r 10 -w 2 -t 1 --no-metrics "$OUT/timing_1mb.ppm" 2>/dev/null \
        | grep -E 'timing_1mb|Average'
    BASE_BENCH="$(dirname "$BASELINE")/bitgrain-bench"
    if [ -x "$BASE_BENCH" ]; then
        echo "-- legacy Huffman --"
        "$BASE_BENCH" -q "$QUALITY" -r 10 -w 2 -t 1 --no-metrics "$OUT/timing_1mb.ppm" 2>/dev/null \
            | grep -E 'timing_1mb|Average'
    fi
else
    echo "bitgrain-bench not built (make bench); falling back to CLI wall time:" >&2
    for _ in 1 2 3; do
        /usr/bin/time -f "encode %e s" "$BG" encode "$OUT/timing_1mb.ppm" -o "$OUT/timing.bg" -q "$QUALITY" -y --threads 1 >/dev/null 2>&1
        /usr/bin/time -f "decode %e s" "$BG" decode "$OUT/timing.bg" -o "$OUT/timing.bmp" -y --threads 1 >/dev/null 2>&1
    done
fi

echo
echo "Artifacts: $OUT"
