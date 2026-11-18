# Bitgrain (.bg) Format Specification

This document specifies the .bg binary format for interoperability. Third-party implementations can read and write .bg files using this specification.

## Overview

Bitgrain is a block-based image codec using 8×8 DCT, quantization, and entropy coding. It supports grayscale (1 channel), RGB (3 channels), and RGBA (4 channels).

## File Structure

```
+----------+----------+----------+------------------+
|  Header  |  Plane 0 |  Plane 1 |  ... (if RGB/RGBA) |
+----------+----------+----------+------------------+
```

## Header (12 bytes)

| Offset | Size | Field    | Description |
|--------|------|----------|-------------|
| 0      | 2    | magic    | `0x42 0x47` ("BG") |
| 2      | 1    | version  | See version table below |
| 3      | 4    | width    | Image width (uint32 LE) |
| 7      | 4    | height   | Image height (uint32 LE) |
| 11     | 1    | quality  | Quantization quality 1–100; 0 means default 50 |

## Block Layout

The image is divided into 8×8 blocks in scan order (left-to-right, top-to-bottom). Blocks in the last row/column may be partially filled; edge pixels are replicated.

- **v1 (grayscale RLE):** One full-resolution plane.
- **v2/v3 (legacy RLE):** 3 or 4 full-resolution planes.
- **v4+ (YCbCr):**
  - RGB profiles: Y full-resolution + Cb/Cr at 4:2:0 (`ceil(w/2) x ceil(h/2)`).
  - RGBA profiles: same Y/Cb/Cr plus full-resolution alpha plane.

## Block Encoding

### RLE path (v1/v2/v3)

Each block is encoded as:

1. **DC coefficient:** 2 bytes, little-endian int16.
2. **AC coefficients:** Repeated (run, level) pairs until EOB.
   - `run`: 1 byte, number of zeros before this coefficient (0–255).
   - `level`: 2 bytes, little-endian int16, the coefficient value.
3. **End-of-block:** `run=0xFF`, `level=0` (3 bytes: `0xFF 0x00 0x00`).

Coefficients are in **zigzag order** (JPEG order):

```
 0  1  8 16  9  2  3 10
17 24 32 25 18 11  4  5
12 19 26 33 40 48 41 34
...
```

### Entropy path (v4+)

Each plane is stored unpadded, immediately followed by the next plane:

```
+-----------------------------+-----------------------------+-----
| u32 LE plane_payload_len    | plane payload (len bytes)   | next
+-----------------------------+-----------------------------+-----
```

The payload is one of two coders. Decoders MUST inspect the first four payload
bytes and dispatch:

- `"ARIT"` → adaptive arithmetic coding, specified in the next section (current
  encoder output).
- Anything else → legacy Huffman coding (specified under *Legacy Huffman
  payload*). New encoders do not emit it, but readers should keep accepting
  stream produced by bitgrain ≤ 2.0.0.

The per-plane length lets a decoder jump exactly to the next plane boundary.

## Adaptive Arithmetic Entropy Payload ("ARIT")

The current encoder codes all coefficients of a plane with a bit-oriented
arithmetic (range) coder driven by adaptive frequency models. The models are
updated from the symbols actually coded, so the code length tracks each
plane's real statistics instead of a fixed Huffman table.

### Payload header

| Offset | Size | Field   | Description |
|--------|------|---------|-------------|
| 0      | 4    | magic   | ASCII `ARIT` (`41 52 49 54`) |
| 4      | 1    | version | Arithmetic payload version = `1` |
| 5      | 1    | config  | bit 0–6: model update period; bit 7: reserved, MUST be 0 |
| 6      | …    | body    | arithmetic-coded bitstream |

### Block tokenisation

Blocks are coded in raster order. For each 8×8 block, coefficients are visited
in zigzag order (`ZIGZAG` below). All symbols go through the same arithmetic
coder; the model chosen depends on the context listed.

**DC (zigzag index 0).** If the stream version uses DC prediction (v10+), the
coded value is `dc - prev_dc` (two's-complement `i16` arithmetic); otherwise it
is `dc`. `prev_dc` starts at 0 and is updated to the current block's decoded
`dc`.

1. **DC category** `c`: `0` when the value is 0, else `1 + floor(log2(|v|))`.
   Coded with a 17-symbol adaptive model, context `min(prev_dc_category, 3)`
   (4 contexts; `prev_dc_category` starts at 0).
2. If `c > 0`: the low `c-1` magnitude bits, as raw bits (`1/2` each).
3. If `c > 0`: the sign bit, with one adaptive binary model per plane.

Magnitude is reconstructed as `(1 << (c-1)) | low_bits`, then the sign applied.

**AC (zigzag indices 1..63).**

1. **Last non-zero index** `L` ∈ `0..63` (`0` = block has no AC). Coded with a
   64-symbol adaptive model, context `bucket(prev_L)` with
   `bucket(L) = 0` for `L=0`, `1` for `L∈1..3`, `2` for `L∈4..15`, `3` otherwise.
2. For `zi` in `1..L`, a **non-zero flag**. Adaptive binary model, context
   `band * 4 + min(run, 3)`, where `band = (zi-1) / 8` and `run` is the number
   of consecutive zero coefficients immediately before `zi` (capped at 3).
   Coefficients at `zi == L` are known non-zero and carry no flag.
3. For each non-zero coefficient (including `zi == L`):
   1. **Magnitude category** `c = floor(log2(|v|))` ∈ `0..15`. Adaptive
      16-symbol model, context `band * 4 + min(prev_c, 3)` where `prev_c` is the
      category of the previous non-zero coefficient in the same block (0 at
      block start).
   2. **Low magnitude bits:** the `c` low bits of `|v|` (the implicit top bit is
      always 1 and is not coded), as raw bits (`1/2` each).
   3. **Sign bit**, adaptive binary model shared by the plane.

### Adaptive frequency models

Every model starts with frequency 1 for each of its `N` symbols (`total = N`).
After a symbol is coded:

- **Update period `P = 0`** (config bits 0–6 = 0): static — frequencies never
  change (uniform code).
- **`P = 1`**: the symbol's frequency is incremented immediately.
- **`P > 1`**: the symbol is counted in a pending bucket; whenever `P` symbols
  have been coded through this model, all pending counts are added and cleared.
- If `total > 32768` after an update, all frequencies are halved with
  `f[i] = (f[i] + 1) / 2` (keeping every frequency ≥ 1 and the total bounded).

Encoder and decoder apply the identical update **after** coding each symbol,
so the model state stays in lock-step; nothing about the models is sent in the
stream beyond `P`.

### Arithmetic coder

Standard 32-bit range coder with deferred underflow:

- State: `low = 0`, `high = 0xFFFFFFFF`, `pending = 0`.
- To code a sub-interval `[cum_low, cum_high)` of `total`:
  `range = high - low + 1`, `high = low + range*cum_high/total - 1`,
  `low = low + range*cum_low/total`, using integer division.
- Renormalise while `high < 0x80000000` (emit 0), `low ≥ 0x80000000`
  (emit 1, subtract `0x80000000`), or `low ≥ 0x40000000` and
  `high < 0xC0000000` (increment `pending`, subtract `0x40000000`); each step
  shifts `low` left and `high = (high << 1) | 1`, both masked to 32 bits.
  Emitting a bit writes it then `pending` copies of its inverse.
- **Raw bits** (probability 1/2) use `half = range >> 1`; a 0-bit sets
  `high = low + half - 1`, a 1-bit sets `low += half`.
- **Decoder** keeps a 32-bit `code` preloaded with the first 32 message bits.
  It finds the symbol with `scaled = ((code - low + 1) * total - 1) / range`,
  selects the symbol whose cumulative interval contains `scaled`, and mirrors
  the same interval/renormalisation steps.

### Tail flush and bit packing

Bits are written MSB-first into bytes with no byte stuffing. At end of plane
the encoder emits one disambiguation bit (0 if `low < 0x40000000`, else 1)
followed by all pending underflow bits, then pads the final partial byte with
zero bits. The decoder reads exactly `plane_payload_len` bytes and treats any
bit past the end as 0; the encoder's zero padding makes this exact.

A plane's payload always contains the arithmetic body even for an empty plane
(`L = 0` for every block still codes one 64-symbol `last_nz` symbol per block);
decoders MUST consume exactly `n_blocks` blocks.

### Choosing the update period

The encoder reads `BITGRAIN_ARITH_UPDATE_PERIOD` (integer, clamped to `0..127`,
default `1`; invalid values fall back to `1` — `0` means static). The chosen
value is written into the payload `config` byte, so decoding never depends on
the environment used for encoding.

## Legacy Huffman payload

Kept for reading streams produced by bitgrain ≤ 2.0.0. Used by v4–v19
JPEG Annex K tables: standard luminance DC table for Y and A,
chrominance DC table for Cb/Cr, and the plain luminance AC table (or the
chrominance AC table for versions that select it) for AC coefficients. The
bitstream is continuous across all blocks of a plane, packed MSB-first, with
`0xFF` stuffed as `0xFF 0x00`, and flushed once per plane padding with 1 bits.


## Quantization

Quality maps to scaled JPEG-like quantization tables (luma and chroma). Newer versions apply increasingly perceptual weighting profiles to close file-size gap versus JPEG.

Base scaling formula:

```
scaled[i] = clamp((base[i] * quality / 100 + 50) / 100, 1, 255)
```

Default table (quality 50):

```
16, 11, 10, 16, 24, 40, 51, 61,
12, 12, 14, 19, 26, 58, 60, 55,
14, 13, 16, 24, 40, 57, 69, 56,
14, 17, 22, 29, 51, 87, 80, 62,
18, 22, 37, 56, 68,109,103, 77,
24, 35, 55, 64, 81,104,113, 92,
49, 64, 78, 87,103,121,120,101,
72, 92, 95, 98,112,100,103, 99
```

Dequantization: `coef[i] *= table[i]` (before IDCT).

## DCT/IDCT

Standard 8×8 DCT-II (forward) and IDCT-II (inverse). Input pixels are centered (0–255 → -128..127). Output coefficients are rounded to int16.

## Optional ICC Trailer

After all plane blocks, an optional trailer may appear:

| Bytes | Field   | Description |
|-------|---------|-------------|
| 3     | magic   | `0x42 0x47 0x78` ("BGx") |
| 1     | type    | Chunk type: 1 = ICC profile |
| 4     | length  | ICC data length (uint32 LE) |
| N     | data    | ICC profile bytes |

Decoders that don't support ICC skip the trailer. Encoders write it only when an ICC profile is provided.

## Extensions (Future)

- Progressive decode / multi-pass refinement (roadmap).

## Reference Implementation

- Encoder/decoder: this repository (Rust core, C FFI).
- C API: `includes/encoder.h` — `bitgrain_encode_*`, `bitgrain_decode`.
