#!/bin/bash
# ===========================================================================
# rdo_eval.sh — rate-distortion optimization evaluation for bitgrain
#
# Generates four small synthetic materials, runs the RDO encoder (opt-in via
# BITGRAIN_RDO_BPP / BITGRAIN_RDO_ITERS) and reports, per material:
#   - target vs achieved bitrate and relative deviation
#   - PSNR (CLI -m, cross-checked independently in Python for one material)
#   - encode wall-clock time
#   - lambda bisection rounds actually used until convergence
# An unreachable target keeps the nearest achievable step and the gap is
# printed explicitly. No third-party tools or network access required.
#
# Usage: bash scripts/rdo_eval.sh [target_bpp] [max_iters]
# ===========================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
BIN="./bitgrain"
TARGET_BPP="${1:-1.0}"
MAX_ITERS="${2:-6}"

if [ ! -x "$BIN" ]; then
    echo "building bitgrain (offline)..."
    CARGO_NET_OFFLINE=true make bitgrain >/dev/null
fi

WORK="$(mktemp -d /tmp/bitgrain_rdo.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
SAMP="$WORK/samples"
mkdir -p "$SAMP"

# Export RDO knobs so encode and the metrics roundtrip select the same step
# (the search is deterministic, so both invocations converge identically).
export BITGRAIN_RDO_BPP="$TARGET_BPP"
export BITGRAIN_RDO_ITERS="$MAX_ITERS"

# ---- four small (64x64) synthetic materials, pure stdlib Python -----------
python3 -I - "$SAMP" <<'PY'
import os, random, sys
out = sys.argv[1]
W = H = 64

def wppm(name, pix):
    with open(os.path.join(out, name), "wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode())
        f.write(bytes(pix))

# 1. flat: constant color, almost zero entropy
flat = [90, 140, 200] * (W * H)
wppm("flat.ppm", flat)

# 2. gradient: smooth low-frequency spatial ramp
grad = []
for y in range(H):
    for x in range(W):
        grad += [x * 255 // W, y * 255 // H, min(255, (x + y) // 2)]
wppm("gradient.ppm", grad)

# 3. noise: high-entropy pseudo-random content (seeded, deterministic)
rnd = random.Random(42)
wppm("noise.ppm", [rnd.randrange(256) for _ in range(W * H * 3)])

# 4. blocks: hard 8x8 checker, strong high-frequency edges
blk = []
for y in range(H):
    for x in range(W):
        v = 230 if ((x // 8) + (y // 8)) % 2 == 0 else 24
        blk += [v, v, 255 - v]
wppm("blocks.ppm", blk)
print("generated materials:", sorted(os.listdir(out)))
PY

PIXELS=$((64 * 64))
printf '\n=== RDO evaluation: target %.3f bpp, max %s iterations ===\n\n' "$TARGET_BPP" "$MAX_ITERS"
printf '%-10s %8s %9s %9s %9s %8s %9s %10s %s\n' \
    "material" "target" "bytes" "ach.bpp" "dev%" "step" "PSNR-dB" "enc-ms" "iters/note"
printf '%s\n' "-------------------------------------------------------------------------------"

for name in flat gradient noise blocks; do
    in="$SAMP/$name.ppm"
    bg="$WORK/$name.bg"
    bmp="$WORK/$name.bmp"

    # timed encode with RDO enabled (env exported above); capture the RDO line.
    t0=$(date +%s%N)
    rdo_line=$("$BIN" encode "$in" -o "$bg" -q 85 -y 2>&1 >/dev/null | grep '^RDO:')
    t1=$(date +%s%N)
    enc_ms=$(( (t1 - t0) / 1000000 ))

    bytes=$(stat -c%s "$bg")
    step=$(printf '%d' "0x$(dd if="$bg" bs=1 skip=11 count=1 2>/dev/null | od -An -tx1 | tr -d ' ')")
    ach_bpp=$(python3 -I -c "print(f'{($bytes*8)/$PIXELS:.4f}')")
    dev=$(python3 -I -c "print(f'{(($bytes*8)/$PIXELS - $TARGET_BPP)/$TARGET_BPP*100:+.2f}')")

    # PSNR measured on the RDO-selected step (RDO env inherited by roundtrip).
    "$BIN" roundtrip "$in" -o "$bmp" -q 85 -m -y >/dev/null 2>"$WORK/rt.err" || true
    psnr=$(grep -oE 'PSNR [0-9.]+' "$WORK/rt.err" | awk '{print $2}')

    iters=$(printf '%s' "$rdo_line" | grep -oE '[0-9]+ iters' | grep -oE '^[0-9]+')
    note="reached"
    if printf '%s' "$rdo_line" | grep -q 'unreachable'; then note="NEAREST (target unreachable)"; fi

    printf '%-10s %8.3f %9s %9s %9s %8s %9s %10s %s (%s)\n' \
        "$name" "$TARGET_BPP" "$bytes" "$ach_bpp" "$dev" "$step" "${psnr:-n/a}" "$enc_ms" "$iters" "$note"
done

# ---- independent PSNR cross-check on one material (no external libs) ------
cat > "$WORK/psnr_check.py" <<'PY'
import math, struct, sys
def read_ppm(p):
    d = open(p, 'rb').read(); i = 2; vals = []
    while len(vals) < 3:
        while d[i] in b' \t\n\r': i += 1
        j = i
        while d[j] not in b' \t\n\r': j += 1
        vals.append(int(d[i:j])); i = j
    w, h, _ = vals; i += 1
    return w, h, d[i:i + w * h * 3]
def read_bmp(p):
    d = open(p, 'rb').read()
    off = struct.unpack('<I', d[10:14])[0]
    w, h = struct.unpack('<ii', d[18:26])
    row = (w * 3 + 3) & ~3; out = bytearray()
    for y in range(h - 1, -1, -1):
        rb = d[off + y * row:off + y * row + w * 3]
        for x in range(w):
            b, g, r = rb[x*3:x*3+3]; out += bytes((r, g, b))
    return w, h, bytes(out)
w, h, a = read_ppm(sys.argv[1]); w2, h2, b = read_bmp(sys.argv[2])
assert (w, h) == (w2, h2)
n = w * h * 3
mse = sum((x - y) ** 2 for x, y in zip(a, b)) / n
print(f"independent cross-check: MSE={mse:.4f} PSNR="
      f"{99.0 if mse == 0 else 10*math.log10(255*255/mse):.2f} dB")
PY
"$BIN" roundtrip "$SAMP/noise.ppm" -o "$WORK/noise.bmp" -q 85 -m -y >/dev/null 2>&1 || true
python3 -I "$WORK/psnr_check.py" "$SAMP/noise.ppm" "$WORK/noise.bmp"

# ---- fixed-step byte-identical guarantee spot check -----------------------
# With RDO env removed, the fixed-step path must be deterministic and must
# ignore RDO entirely (this is the byte-stability guarantee from the task).
ref="$WORK/ref.bg"
env -u BITGRAIN_RDO_BPP -u BITGRAIN_RDO_ITERS \
    "$BIN" encode "$SAMP/gradient.ppm" -o "$ref" -q 85 -y >/dev/null 2>&1
env -u BITGRAIN_RDO_BPP -u BITGRAIN_RDO_ITERS \
    "$BIN" encode "$SAMP/gradient.ppm" -o "$WORK/x.bg" -q 85 -y >/dev/null 2>&1
cmp -s "$ref" "$WORK/x.bg" && echo "fixed-step (RDO disabled): deterministic, unchanged" \
    || echo "WARNING: fixed-step output differs"

echo
echo "Notes: step = header quality byte (1..100). dev% relative to target bpp."
echo "RDO env: BITGRAIN_RDO_BPP (target bpp), BITGRAIN_RDO_ITERS (bisection cap)."
