#!/usr/bin/env python3
"""Wavelet transform-domain mode (v20/v21) evaluation for bitgrain.

Self-contained: generates 4 small synthetic samples (deterministic, seeded),
then measures, per sample and per mode (spatial DCT baseline / haar / cdf53):

  - .bg size and size ratio (vs raw RGB and vs spatial .bg)
  - PSNR of the full encode->decode round trip
  - encode / decode wall time (median, in-process via the C API)
  - subband energy distribution (LL/LH/HL/HH, from the encoder itself)

Also checks:
  - spatial-path byte identity vs golden hashes (original DCT encoding must
    be byte-for-byte unchanged by the wavelet patch)
  - invalid transform sizes are rejected with an error (encode and decode)
  - odd dimensions round-trip legally
  - CLI integration: `bitgrain encode` / `bitgrain roundtrip` honor
    BITGRAIN_WAVELET and emit v20 streams

No third-party dependencies; talks to rust/target/release/libbitgrain.so
via ctypes (the public C API from includes/encoder.h). Run from repo root:

    python3 tests/wavelet_eval.py
"""
import ctypes
import hashlib
import math
import os
import random
import re
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "wavelet_out"
# Self-contained shared lib (rust cdylib + dct/quant PIC objects), same code
# the CLI links statically. Built by `make libbitgrain-shared`.
SO = ROOT / "build" / "lib" / "libbitgrain.so"
CLI = ROOT / "bitgrain"
QUALITY = 85
REPS = 20

# Golden sha256 of the spatial (default, BITGRAIN_WAVELET unset) .bg output
# for each generated sample, captured with the unmodified 2.0.0 encoder on
# this machine. The wavelet patch must not change the spatial path at all.
GOLDEN_SPATIAL_SHA256 = {
    "gradient.ppm": "6619f18eca32820e168b2a7f4ba88e01e6724f2d70ef710dc910133459bd1c98",  # 987 B, v18
    "noise.ppm": "0796cb02c48e0e09b515ad09b1c94477701c4d6394ee97a965ec5cc41926cb47",  # 5181 B, v18
    "edges.ppm": "64aa8b9ead30c783f218fabce9543c2bed8b217c11e19c9b742918a3b8933b04",  # 1472 B, v18
    "texture.ppm": "1591a2b58de048f52e865948fa852e41f43725306eab3e951cce7033aaf45b0e",  # 1804 B, v18
}

# ---------------------------------------------------------------------------
# Sample generation (deterministic, small, 4 material classes)
# ---------------------------------------------------------------------------

def gen_gradient(w=128, h=128):
    """Smooth continuous-tone gradients."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            r = x * 255 // (w - 1)
            g = y * 255 // (h - 1)
            b = (x + y) * 255 // (w + h - 2)
            px += bytes((r, g, b))
    return w, h, bytes(px)

def gen_noise(w=128, h=128, seed=1234):
    """Uniform random RGB noise (incompressible worst case)."""
    rng = random.Random(seed)
    return w, h, rng.randbytes(w * h * 3)

def gen_edges(w=128, h=128):
    """Hard-edged graphics: flat fills, bars, a disc, a checker patch."""
    px = bytearray(b"\xf0" * (w * h * 3))  # light background
    def put(x, y, rgb):
        i = (y * w + x) * 3
        px[i:i+3] = bytes(rgb)
    # black rectangle
    for y in range(16, 48):
        for x in range(16, 80):
            put(x, y, (10, 10, 10))
    # red disc
    for y in range(h):
        for x in range(w):
            if (x - 96) ** 2 + (y - 96) ** 2 <= 24 ** 2:
                put(x, y, (220, 30, 30))
    # blue horizontal bars
    for y in range(64, 68):
        for x in range(0, w):
            put(x, y, (20, 40, 230))
    for y in range(72, 74):
        for x in range(0, w):
            put(x, y, (20, 40, 230))
    # green vertical bar
    for y in range(8, 120):
        for x in range(100, 106):
            put(x, y, (30, 180, 60))
    # 16x16 checker patch (1px cells)
    for y in range(56, 72):
        for x in range(56, 72):
            if (x + y) % 2 == 0:
                put(x, y, (0, 0, 0))
    return w, h, bytes(px)

def gen_texture(w=127, h=97):
    """Photo-like mixed texture: sinusoids + gradient + light noise (odd dims)."""
    rng = random.Random(99)
    px = bytearray()
    for y in range(h):
        for x in range(w):
            r = 128 + 60 * math.sin(x * 0.31) * math.cos(y * 0.23) + x * 0.4
            g = 128 + 50 * math.sin((x + y) * 0.17) + y * 0.3
            b = 100 + 70 * math.cos(x * 0.11 + y * 0.07)
            n = rng.randint(-6, 6)
            px += bytes((
                max(0, min(255, int(r) + n)),
                max(0, min(255, int(g) + n)),
                max(0, min(255, int(b))),
            ))
    return w, h, bytes(px)

GENERATORS = {
    "gradient.ppm": gen_gradient,
    "noise.ppm": gen_noise,
    "edges.ppm": gen_edges,
    "texture.ppm": gen_texture,
}

def write_ppm(path, w, h, rgb):
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w, h))
        f.write(rgb)

# ---------------------------------------------------------------------------
# C API via ctypes
# ---------------------------------------------------------------------------

def load_api():
    lib = ctypes.CDLL(str(SO))
    u8p = ctypes.c_void_p
    lib.bitgrain_encode_rgb.restype = ctypes.c_int
    lib.bitgrain_encode_rgb.argtypes = [u8p, ctypes.c_uint32, ctypes.c_uint32,
                                        u8p, ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_int32), ctypes.c_uint8]
    lib.bitgrain_decode.restype = ctypes.c_int
    lib.bitgrain_decode.argtypes = [u8p, ctypes.c_int32, u8p, ctypes.c_uint32,
                                    ctypes.POINTER(ctypes.c_uint32),
                                    ctypes.POINTER(ctypes.c_uint32),
                                    ctypes.POINTER(ctypes.c_uint32)]
    lib.bitgrain_last_error_message.restype = ctypes.c_char_p
    lib.bitgrain_last_error_message.argtypes = []
    lib.bitgrain_clear_error.argtypes = []
    return lib

def set_mode(mode):
    if mode == "spatial":
        os.environ.pop("BITGRAIN_WAVELET", None)
    else:
        os.environ["BITGRAIN_WAVELET"] = mode

def encode_once(lib, rgb, w, h, quality=QUALITY):
    cap = w * h * 3 * 2 + 4096
    out = ctypes.create_string_buffer(cap)
    out_len = ctypes.c_int32(0)
    lib.bitgrain_clear_error()
    ret = lib.bitgrain_encode_rgb(rgb, w, h, out, cap, ctypes.byref(out_len), quality)
    if ret != 0:
        return None
    return out.raw[:out_len.value]

def decode_once(lib, bg):
    # The decoder writes into the caller's buffer and the colorspace
    # conversion iterates over it, so the buffer must be exactly w*h*ch.
    if len(bg) < 12 or bg[0:2] != b"BG":
        return None
    ver = bg[2]
    w = struct.unpack_from("<I", bg, 3)[0]
    h = struct.unpack_from("<I", bg, 7)[0]
    if ver == 1:
        ch = 1
    elif ver == 2 or ver == 20:
        ch = 3
    elif ver == 3 or ver == 21:
        ch = 4
    elif 4 <= ver <= 19:
        ch = 3 if ver % 2 == 0 else 4
    else:
        ch = 3
    cap = w * h * ch
    out = ctypes.create_string_buffer(cap)
    w_ = ctypes.c_uint32(0)
    h_ = ctypes.c_uint32(0)
    ch_ = ctypes.c_uint32(0)
    lib.bitgrain_clear_error()
    ret = lib.bitgrain_decode(bg, len(bg), out, cap,
                              ctypes.byref(w_), ctypes.byref(h_), ctypes.byref(ch_))
    if ret != 0:
        return None
    return w_.value, h_.value, ch_.value, out.raw[:w_.value * h_.value * ch_.value]

def timed(fn, reps=REPS):
    best = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        r = fn()
        best.append(time.perf_counter_ns() - t0)
    return statistics.median(best) / 1e6, r  # ms, last result

def psnr(a, b):
    n = len(a)
    assert n == len(b)
    se = 0
    for i in range(0, n, 4096):
        for x, y in zip(a[i:i+4096], b[i:i+4096]):
            d = x - y
            se += d * d
    mse = se / n
    if mse == 0:
        return float("inf")
    return 10 * math.log10(255.0 * 255.0 / mse)

def encode_with_stats(lib, rgb, w, h):
    """Encode with BITGRAIN_WAVELET_STATS=1, capturing the encoder's stderr."""
    os.environ["BITGRAIN_WAVELET_STATS"] = "1"
    rfd, wfd = os.pipe()
    saved = os.dup(2)
    os.dup2(wfd, 2)
    try:
        bg = encode_once(lib, rgb, w, h)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(wfd)
        os.environ.pop("BITGRAIN_WAVELET_STATS", None)
    chunks = []
    while True:
        data = os.read(rfd, 65536)
        if not data:
            break
        chunks.append(data)
    os.close(rfd)
    return bg, b"".join(chunks).decode(errors="replace")

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

FAILURES = []

def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)

def main():
    # Build everything (cargo is incremental; make relinks CLI + shared lib).
    for cmd in (["cargo", "build", "--release"],):
        r = subprocess.run(cmd, cwd=ROOT / "rust", capture_output=True)
        if r.returncode != 0:
            print(f"error: {' '.join(cmd)} failed:\n{r.stderr.decode()}", file=sys.stderr)
            return 1
    r = subprocess.run(["make", "bitgrain", "libbitgrain-shared"], cwd=ROOT,
                       capture_output=True)
    if r.returncode != 0:
        print(f"error: make failed:\n{r.stderr.decode()}", file=sys.stderr)
        return 1
    if not SO.exists():
        print(f"error: {SO} not found after build", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    lib = load_api()

    # 1. generate samples
    samples = {}
    for name, gen in GENERATORS.items():
        w, h, rgb = gen()
        samples[name] = (w, h, rgb)
        write_ppm(OUT / name, w, h, rgb)
    print(f"samples: {', '.join(f'{n} ({samples[n][0]}x{samples[n][1]})' for n in samples)}")

    # 2. spatial byte identity vs golden
    print("\n== spatial path byte identity (BITGRAIN_WAVELET unset) ==")
    set_mode("spatial")
    spatial_bg = {}
    for name, (w, h, rgb) in samples.items():
        bg = encode_once(lib, rgb, w, h)
        spatial_bg[name] = bg
        digest = hashlib.sha256(bg).hexdigest()
        golden = GOLDEN_SPATIAL_SHA256.get(name, "")
        check(f"{name} spatial .bg identical to 2.0.0 baseline",
              golden not in ("", "FILL") and digest == golden,
              f"sha256={digest[:16]}… size={len(bg)}")
        if bg[2] != 18:
            check(f"{name} spatial version byte == 18", False, f"got {bg[2]}")

    # 3. metrics per sample per mode
    print("\n== metrics (quality %d, median of %d reps) ==" % (QUALITY, REPS))
    modes = ["spatial", "haar", "cdf53"]
    results = {}  # (sample, mode) -> dict
    for name, (w, h, rgb) in samples.items():
        raw = w * h * 3
        for mode in modes:
            set_mode(mode)
            enc_ms, bg = timed(lambda: encode_once(lib, rgb, w, h))
            if bg is None:
                check(f"{name}/{mode} encode", False, "encode returned error")
                continue
            dec_ms, dec = timed(lambda: decode_once(lib, bg))
            if dec is None:
                check(f"{name}/{mode} decode", False, "decode returned error")
                continue
            dw, dh, ch, px = dec
            ok_dims = (dw, dh, ch) == (w, h, 3)
            p = psnr(rgb, px) if ok_dims else float("nan")
            results[(name, mode)] = dict(bg=bg, size=len(bg), raw=raw,
                                         enc_ms=enc_ms, dec_ms=dec_ms, psnr=p)
            if not ok_dims:
                check(f"{name}/{mode} dims", False, f"got {dw}x{dh}x{ch}")
    set_mode("spatial")

    hdr = f"{'sample':<14}{'mode':<9}{'raw B':>8}{'.bg B':>9}{'vs raw':>8}{'vs spatial':>11}{'PSNR dB':>9}{'enc ms':>9}{'dec ms':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in samples:
        sp = results.get((name, "spatial"))
        for mode in modes:
            r = results.get((name, mode))
            if not r:
                continue
            vs_spatial = r["size"] / sp["size"] if sp else float("nan")
            print(f"{name:<14}{mode:<9}{r['raw']:>8}{r['size']:>9}"
                  f"{r['size']/r['raw']:>8.3f}{vs_spatial:>11.3f}"
                  f"{r['psnr']:>9.2f}{r['enc_ms']:>9.3f}{r['dec_ms']:>9.3f}")

    # 4. subband energy distribution (from the encoder, stderr stats line)
    print("\n== subband energy distribution (luma, one-level 2D DWT) ==")
    energy_ok = True
    for name, (w, h, rgb) in samples.items():
        for mode in ("haar", "cdf53"):
            set_mode(mode)
            bg, stats = encode_with_stats(lib, rgb, w, h)
            m = re.search(
                r"energy LL=([\d.]+)% LH=([\d.]+)% HL=([\d.]+)% HH=([\d.]+)%", stats)
            if not m:
                check(f"{name}/{mode} stats line", False, f"stderr={stats!r}")
                energy_ok = False
                continue
            ll, lh, hl, hh = (float(g) for g in m.groups())
            print(f"  {name:<14}{mode:<7} LL={ll:6.2f}%  LH={lh:5.2f}%  "
                  f"HL={hl:5.2f}%  HH={hh:5.2f}%   ({stats.strip()})")
            if abs(ll + lh + hl + hh - 100.0) > 0.5:
                check(f"{name}/{mode} energy sums to 100%", False)
                energy_ok = False
    set_mode("spatial")

    # 5. invalid transform sizes must error
    print("\n== invalid transform size handling ==")
    for mode in ("haar", "cdf53"):
        set_mode(mode)
        for (bw, bh) in ((1, 1), (1, 64), (64, 1)):
            lib.bitgrain_clear_error()
            cap = 8192
            out = ctypes.create_string_buffer(cap)
            out_len = ctypes.c_int32(0)
            ret = lib.bitgrain_encode_rgb(b"\x80" * (bw * bh * 3), bw, bh,
                                          out, cap, ctypes.byref(out_len), QUALITY)
            msg = (lib.bitgrain_last_error_message() or b"").decode()
            check(f"encode {bw}x{bh} ({mode}) rejected",
                  ret != 0 and "wavelet" in msg, f"ret={ret} msg={msg!r}")
    # hand-crafted v20 stream with illegal 1x1 transform size must fail decode
    bad = b"BG" + bytes([20]) + struct.pack("<II", 1, 1) + bytes([QUALITY, 0]) + b"\x00" * 16
    lib.bitgrain_clear_error()
    ret = decode_once(lib, bad)
    check("decode v20 1x1 rejected", ret is None, f"result={ret}")
    # corrupt basis tag must fail decode
    good = results.get(("gradient.ppm", "haar"), {}).get("bg")
    if good:
        bad2 = bytearray(good)
        bad2[12] = 9  # invalid basis tag
        lib.bitgrain_clear_error()
        ret = decode_once(lib, bytes(bad2))
        check("decode invalid basis tag rejected", ret is None, f"result={ret}")
    set_mode("spatial")

    # 6. odd dimensions are legal and round-trip
    print("\n== odd-size legality ==")
    tw, th, _ = samples["texture.ppm"]
    check("texture sample has odd height", th % 2 == 1, f"{tw}x{th}")
    sp_psnr = results.get(("texture.ppm", "spatial"), {}).get("psnr", 0)
    for mode in ("haar", "cdf53"):
        r = results.get(("texture.ppm", mode))
        # Wavelet PSNR should be within 1 dB of the spatial path on the
        # same odd-size sample (absolute PSNR is content-dependent).
        check(f"odd-size roundtrip ({mode})",
              r is not None and r["psnr"] > sp_psnr - 1.0,
              f"PSNR={r['psnr'] if r else 'n/a'} vs spatial {sp_psnr:.2f}")

    # 7. CLI integration (encode + roundtrip honor the env var; v20 emitted)
    print("\n== CLI integration ==")
    if CLI.exists():
        env = dict(os.environ, BITGRAIN_WAVELET="haar")
        cli_bg = OUT / "cli_wavelet.bg"
        r = subprocess.run([str(CLI), "encode", str(OUT / "gradient.ppm"),
                            "-o", str(cli_bg), "-y"],
                           env=env, capture_output=True)
        ver = cli_bg.read_bytes()[2] if cli_bg.exists() else None
        check("CLI encode emits v20 under BITGRAIN_WAVELET=haar",
              r.returncode == 0 and ver == 20, f"rc={r.returncode} version={ver}")
        r = subprocess.run([str(CLI), "roundtrip", str(OUT / "gradient.ppm"),
                            "-o", str(OUT / "cli_wavelet_rt.png"), "-m", "-y"],
                           env=env, capture_output=True)
        check("CLI roundtrip works in wavelet mode",
              r.returncode == 0 and (OUT / "cli_wavelet_rt.png").exists(),
              f"rc={r.returncode} stderr={r.stderr.decode()[-200:]}")
        r = subprocess.run([str(CLI), "encode", str(OUT / "gradient.ppm"),
                            "-o", str(OUT / "cli_spatial.bg"), "-y"],
                           capture_output=True)
        ver = (OUT / "cli_spatial.bg").read_bytes()[2]
        check("CLI encode defaults to spatial v18 (env unset)",
              r.returncode == 0 and ver == 18, f"rc={r.returncode} version={ver}")
    else:
        print("  (CLI binary not found, skipping)")

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else
                  f"FAILURES: {', '.join(FAILURES)}"))
    return 0 if not FAILURES else 1

if __name__ == "__main__":
    sys.exit(main())
