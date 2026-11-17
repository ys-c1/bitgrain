use crate::bitstream;
use crate::block::Block;
use crate::blockizer::Blockizer;
use crate::colorspace;
use crate::dct;
use crate::entropy;
#[cfg(not(test))]
use crate::ffi::quantize_block;
#[cfg(not(test))]
use crate::ffi::dequantize_block;
use crate::huffman;
use crate::zigzag::ZIGZAG;
use rayon::prelude::*;
const BLOCK_TILE_SIZE: usize = 512;
const PARALLEL_BLOCKS_THRESHOLD: usize = 384;
const PARALLEL_PLANE_PIXELS_THRESHOLD: usize = 262_144;
const PARALLEL_IMAGE_PIXELS_THRESHOLD: usize = 262_144;
const PARALLEL_DEQUANT_BLOCKS_THRESHOLD: usize = 384;
const PARALLEL_DEQUANT_PIXELS_THRESHOLD: usize = 262_144;

#[inline]
fn should_parallel_dequant(n_blocks: usize, w: usize, h: usize) -> bool {
    n_blocks >= PARALLEL_DEQUANT_BLOCKS_THRESHOLD
        && w.saturating_mul(h) >= PARALLEL_DEQUANT_PIXELS_THRESHOLD
}

/// Write one reconstructed block back into a flat plane buffer, replicating
/// edge pixels for partial blocks at the image border.
#[inline]
fn write_block_to_plane(block: &Block, plane: &mut [u8], w: usize, h: usize, bx: usize, by: usize) {
    if bx + 8 <= w && by + 8 <= h {
        for y in 0..8 {
            let dst = (by + y) * w + bx;
            let src = y * 8;
            for x in 0..8 {
                plane[dst + x] = (block.data[src + x] + 128).clamp(0, 255) as u8;
            }
        }
        return;
    }
    for y in 0..8 {
        for x in 0..8 {
            let py = by + y;
            let px = bx + x;
            if py < h && px < w {
                plane[py * w + px] = (block.data[y * 8 + x] + 128).clamp(0, 255) as u8;
            }
        }
    }
}

#[inline]
fn should_parallel_blocks(n_blocks: usize, plane_w: usize, plane_h: usize) -> bool {
    n_blocks >= PARALLEL_BLOCKS_THRESHOLD
        && plane_w.saturating_mul(plane_h) >= PARALLEL_PLANE_PIXELS_THRESHOLD
}

#[inline]
fn should_parallel_planes(width: usize, height: usize) -> bool {
    width.saturating_mul(height) >= PARALLEL_IMAGE_PIXELS_THRESHOLD
}

/// .bg header: "BG" + version + width(u32 LE) + height(u32 LE) + quality(u8) = 12 bytes.
///
/// Version byte:
///   1 = grayscale,  RLE entropy
///   2 = RGB planar, RLE entropy   (legacy)
///   3 = RGBA planar,RLE entropy   (legacy)
///   4 = YCbCr 4:2:0, Huffman     (RGB input, best compression)
///   5 = YCbCr 4:2:0 + A, Huffman (RGBA input)
///   6 = YCbCr 4:2:0, Huffman + chroma AC table (RGB input, improved size)
///   7 = YCbCr 4:2:0 + A, Huffman + chroma AC table (RGBA input, improved size)
///   8 = YCbCr 4:2:0, perceptual quant + chroma AC (RGB input, better size)
///   9 = YCbCr 4:2:0 + A, perceptual quant + chroma AC (RGBA input, better size)
///  10 = YCbCr 4:2:0, perceptual quant + chroma AC + DC delta (JPEG-like)
///  11 = YCbCr 4:2:0 + A, perceptual quant + chroma AC + DC delta (JPEG-like)
///  12 = YCbCr 4:2:0, stronger perceptual quant + chroma AC + DC delta
///  13 = YCbCr 4:2:0 + A, stronger perceptual quant + chroma AC + DC delta
///  14 = YCbCr 4:2:0, aggressive perceptual quant + chroma AC + DC delta
///  15 = YCbCr 4:2:0 + A, aggressive perceptual quant + chroma AC + DC delta
///  16 = YCbCr 4:2:0, very aggressive perceptual quant + chroma AC + DC delta
///  17 = YCbCr 4:2:0 + A, very aggressive perceptual quant + chroma AC + DC delta
///  18 = YCbCr 4:2:0, ultra perceptual + AC sparsify + chroma AC + DC delta
///  19 = YCbCr 4:2:0 + A, ultra perceptual + AC sparsify + chroma AC + DC delta
pub const BG_HEADER_SIZE: usize = 3 + 4 + 4 + 1;

const BG_MAGIC_GRAY:    &[u8; 3] = b"BG\x01";
const BG_MAGIC_RGB:     &[u8; 3] = b"BG\x02";
const BG_MAGIC_RGBA:    &[u8; 3] = b"BG\x03";
const BG_MAGIC_YUV420_V8:  &[u8; 3] = b"BG\x12";
const BG_MAGIC_YUV420A_V8: &[u8; 3] = b"BG\x13";

/// Standard JPEG luminance quantization table (quality ~50).
pub fn default_quant_table() -> [i16; 64] {
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ]
}

/// Standard JPEG chrominance quantization table (quality ~50).
pub fn default_chroma_quant_table() -> [i16; 64] {
    [
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ]
}

/// Scale a quant table by quality using the standard JPEG formula.
/// quality 50 = default table; 100 = minimal quantization; 1 = heavy.
pub fn quant_table_for_quality(quality: u8) -> [i16; 64] {
    scale_quant_table(&default_quant_table(), quality)
}

pub fn chroma_quant_table_for_quality(quality: u8) -> [i16; 64] {
    scale_quant_table(&default_chroma_quant_table(), quality)
}

fn scale_quant_table(base: &[i16; 64], quality: u8) -> [i16; 64] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    let mut out = [0i16; 64];
    for i in 0..64 {
        let v = (base[i] as i32 * scale + 50) / 100;
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

fn scale_quant_table_perceptual(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Conservative perceptual tuning:
    // - stronger attenuation on chroma detail
    // - progressively stronger quantization toward high frequencies
    let mut out = scale_quant_table(base, quality);
    // 8x8 weights (percent). Top-left keeps detail; bottom-right compresses harder.
    const LUMA_W: [i32; 64] = [
        100,100,100,102,104,106,108,110,
        100,100,102,104,106,108,110,112,
        100,102,104,106,108,110,114,118,
        102,104,106,108,110,114,118,122,
        104,106,108,110,114,118,122,128,
        106,108,110,114,118,122,128,134,
        108,110,114,118,122,128,134,142,
        110,112,118,122,128,134,142,152,
    ];
    const CHROMA_W: [i32; 64] = [
        112,114,116,120,124,128,132,136,
        114,116,120,124,128,132,136,142,
        116,120,124,128,132,136,142,148,
        120,124,128,132,136,142,148,156,
        124,128,132,136,142,148,156,166,
        128,132,136,142,148,156,166,178,
        132,136,142,148,156,166,178,192,
        136,142,148,156,166,178,192,208,
    ];
    let w = if chroma { &CHROMA_W } else { &LUMA_W };
    for i in 0..64 {
        let v = (out[i] as i32 * w[i] + 50) / 100;
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual(&default_chroma_quant_table(), quality, true)
}

fn scale_quant_table_perceptual_v2(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Stronger than v1 perceptual profile, still bounded and deterministic.
    let mut out = scale_quant_table(base, quality);
    const LUMA_W2: [i32; 64] = [
        100,100,102,104,108,112,116,122,
        100,102,104,108,112,116,122,128,
        102,104,108,112,116,122,130,138,
        104,108,112,116,122,130,138,148,
        108,112,116,122,130,138,148,160,
        112,116,122,130,138,148,160,174,
        116,122,130,138,148,160,174,190,
        122,128,138,148,160,174,190,208,
    ];
    const CHROMA_W2: [i32; 64] = [
        125,130,136,144,152,162,174,188,
        130,136,144,152,162,174,188,204,
        136,144,152,162,174,188,204,222,
        144,152,162,174,188,204,222,242,
        152,162,174,188,204,222,242,255,
        162,174,188,204,222,242,255,255,
        174,188,204,222,242,255,255,255,
        188,204,222,242,255,255,255,255,
    ];
    let w = if chroma { &CHROMA_W2 } else { &LUMA_W2 };
    for i in 0..64 {
        let mut v = (out[i] as i32 * w[i] + 50) / 100;
        // Slightly more pressure at medium/low quality to close size gap.
        if quality <= 85 {
            if chroma {
                v = (v * 106 + 50) / 100;
            } else if i >= 32 {
                v = (v * 103 + 50) / 100;
            }
        }
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual_v2(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v2(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual_v2(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v2(&default_chroma_quant_table(), quality, true)
}

fn scale_quant_table_perceptual_v3(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Experimental aggressive profile to probe size-gap closure.
    let mut out = scale_quant_table(base, quality);
    const LUMA_W3: [i32; 64] = [
        102,104,108,112,118,126,136,148,
        104,108,112,118,126,136,148,162,
        108,112,118,126,136,148,162,178,
        112,118,126,136,148,162,178,196,
        118,126,136,148,162,178,196,216,
        126,136,148,162,178,196,216,238,
        136,148,162,178,196,216,238,255,
        148,162,178,196,216,238,255,255,
    ];
    const CHROMA_W3: [i32; 64] = [
        145,154,166,180,196,214,234,255,
        154,166,180,196,214,234,255,255,
        166,180,196,214,234,255,255,255,
        180,196,214,234,255,255,255,255,
        196,214,234,255,255,255,255,255,
        214,234,255,255,255,255,255,255,
        234,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
    ];
    let w = if chroma { &CHROMA_W3 } else { &LUMA_W3 };
    for i in 0..64 {
        let mut v = (out[i] as i32 * w[i] + 50) / 100;
        if quality <= 92 {
            if chroma {
                v = (v * 112 + 50) / 100;
            } else if i >= 24 {
                v = (v * 106 + 50) / 100;
            }
        }
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual_v3(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v3(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual_v3(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v3(&default_chroma_quant_table(), quality, true)
}

fn scale_quant_table_perceptual_v4(base: &[i16; 64], quality: u8, chroma: bool) -> [i16; 64] {
    // Very aggressive profile for compression experiments.
    let mut out = scale_quant_table(base, quality);
    const LUMA_W4: [i32; 64] = [
        106,112,120,130,142,156,172,190,
        112,120,130,142,156,172,190,210,
        120,130,142,156,172,190,210,232,
        130,142,156,172,190,210,232,255,
        142,156,172,190,210,232,255,255,
        156,172,190,210,232,255,255,255,
        172,190,210,232,255,255,255,255,
        190,210,232,255,255,255,255,255,
    ];
    const CHROMA_W4: [i32; 64] = [
        170,184,200,218,238,255,255,255,
        184,200,218,238,255,255,255,255,
        200,218,238,255,255,255,255,255,
        218,238,255,255,255,255,255,255,
        238,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,
    ];
    let w = if chroma { &CHROMA_W4 } else { &LUMA_W4 };
    for i in 0..64 {
        let mut v = (out[i] as i32 * w[i] + 50) / 100;
        if quality <= 95 {
            if chroma {
                v = (v * 118 + 50) / 100;
            } else if i >= 20 {
                v = (v * 109 + 50) / 100;
            }
        }
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

pub fn quant_table_for_quality_perceptual_v4(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v4(&default_quant_table(), quality, false)
}

pub fn chroma_quant_table_for_quality_perceptual_v4(quality: u8) -> [i16; 64] {
    scale_quant_table_perceptual_v4(&default_chroma_quant_table(), quality, true)
}

fn build_sparsify_thresholds(quality: u8, is_chroma: bool) -> [i16; 64] {
    // Precompute once per plane to avoid per-block branchy threshold math.
    let q = quality.clamp(1, 100);
    let base_t: i16 = if is_chroma {
        if q >= 90 { 2 } else if q >= 80 { 3 } else { 4 }
    } else if q >= 90 {
        1
    } else if q >= 80 {
        2
    } else {
        3
    };
    let mut thr = [0i16; 64];
    for zi in 1..64 {
        thr[zi] = if zi >= 56 {
            base_t + 2
        } else if zi >= 40 {
            base_t + 1
        } else if zi >= 24 {
            base_t
        } else {
            base_t.saturating_sub(1).max(1)
        };
    }
    thr
}

#[inline]
fn sparsify_ac_block(block: &mut Block, thresholds: &[i16; 64]) {
    for zi in 1..64 {
        let z = ZIGZAG[zi];
        if block.data[z].abs() <= thresholds[zi] {
            block.data[z] = 0;
        }
    }
}

/// Scalar quantization matching the production SSE2/AVX2 kernel: float
/// division rounded to nearest-even (see c/quant.c). Used in test builds,
/// where the C FFI symbols are not linked.
#[cfg(test)]
#[inline]
fn scalar_quantize(block: &mut [i16; 64], table: &[i16; 64]) {
    for i in 0..64 {
        block[i] = ((block[i] as f32) / (table[i] as f32)).round_ties_even() as i16;
    }
}

/// Scalar dequantization matching `dequantize_block_scalar_impl` in c/quant.c.
#[cfg(test)]
#[inline]
fn scalar_dequantize(block: &mut [i16; 64], table: &[i16; 64]) {
    for i in 0..64 {
        let v = block[i] as i32 * table[i] as i32;
        block[i] = v.clamp(-32768, 32767) as i16;
    }
}

#[inline]
pub fn quantize(block: &mut [i16; 64], table: &[i16; 64]) {
    #[cfg(not(test))]
    unsafe { quantize_block(block.as_mut_ptr(), table.as_ptr()); }
    #[cfg(test)]
    scalar_quantize(block, table);
}

/// Dequantize one block: C SIMD kernel in release, scalar in test builds.
#[inline]
fn dequantize(block: &mut Block, table: &[i16; 64]) {
    #[cfg(not(test))]
    unsafe { dequantize_block(block.data.as_mut_ptr(), table.as_ptr()); }
    #[cfg(test)]
    scalar_dequantize(&mut block.data, table);
}

fn write_header(out: &mut [u8], pos: &mut i32, magic: &[u8; 3], w: usize, h: usize, q: u8) {
    if (*pos as usize) + BG_HEADER_SIZE > out.len() { return; }
    bitstream::write_bytes(out, pos, magic);
    for b in (w as u32).to_le_bytes() { bitstream::write_byte(out, pos, b); }
    for b in (h as u32).to_le_bytes() { bitstream::write_byte(out, pos, b); }
    bitstream::write_byte(out, pos, if q == 0 { 50 } else { q });
}

fn write_icc_trailer(out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>) {
    let Some(data) = icc else { return };
    if data.is_empty() { return; }
    let need = 3 + 1 + 4 + data.len();
    if (*pos as usize) + need > out.len() { return; }
    bitstream::write_byte(out, pos, b'B');
    bitstream::write_byte(out, pos, b'G');
    bitstream::write_byte(out, pos, b'x');
    bitstream::write_byte(out, pos, 1);
    for b in (data.len() as u32).to_le_bytes() { bitstream::write_byte(out, pos, b); }
    bitstream::write_bytes(out, pos, data);
}

// ---------------------------------------------------------------------------
// RLE path (legacy v1/v2/v3)
// ---------------------------------------------------------------------------

fn encode_blocks_rle(
    blocks: &mut [Block],
    table: &[i16; 64],
    out: &mut [u8],
    pos: &mut i32,
    plane_w: usize,
    plane_h: usize,
) {
    if should_parallel_blocks(blocks.len(), plane_w, plane_h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dct::dct(block);
                quantize(&mut block.data, table);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            dct::dct(block);
            quantize(&mut block.data, table);
        }
    }
    for block in blocks.iter() {
        entropy::encode_block_to_buffer(block, out, pos);
    }
}

fn encode_channel_rle(blocks: &mut [Block], table: &[i16; 64], plane_w: usize, plane_h: usize) -> Vec<u8> {
    if should_parallel_blocks(blocks.len(), plane_w, plane_h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dct::dct(block);
                quantize(&mut block.data, table);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            dct::dct(block);
            quantize(&mut block.data, table);
        }
    }
    let cap = blocks.len() * (2 + 63 * 3 + 3);
    let mut buf = vec![0u8; cap];
    let mut p: i32 = 0;
    for block in blocks.iter() {
        entropy::encode_block_to_buffer(block, &mut buf, &mut p);
    }
    buf.truncate(p as usize);
    buf
}

pub fn encode_grayscale(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32,
) {
    write_header(out, pos, BG_MAGIC_GRAY, width, height, quality);
    let table = quant_table_for_quality(quality);
    let blockizer = Blockizer::new(width, height);
    let mut blocks = blockizer.generate_blocks(image);
    encode_blocks_rle(&mut blocks, &table, out, pos, width, height);
}

// ---------------------------------------------------------------------------
// Huffman + YCbCr 4:2:0 path (v4/v5) — best compression
// ---------------------------------------------------------------------------

/// Encode blocks with Huffman into a Vec<u8>. Parallel DCT+quant, sequential Huffman.
fn encode_channel_huffman(
    blocks: &mut [Block],
    table: &[i16; 64],
    plane_w: usize,
    plane_h: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    sparsify_thresholds: Option<&[i16; 64]>,
) -> Vec<u8> {
    if should_parallel_blocks(blocks.len(), plane_w, plane_h) {
        blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
            for block in chunk.iter_mut() {
                dct::dct(block);
                quantize(&mut block.data, table);
                if let Some(thr) = sparsify_thresholds {
                    sparsify_ac_block(block, thr);
                }
                huffman::clamp_block_jpeg_coeffs(block);
            }
        });
    } else {
        for block in blocks.iter_mut() {
            dct::dct(block);
            quantize(&mut block.data, table);
            if let Some(thr) = sparsify_thresholds {
                sparsify_ac_block(block, thr);
            }
            huffman::clamp_block_jpeg_coeffs(block);
        }
    }
    huffman::encode_plane_with_profile(blocks, is_chroma, use_chroma_ac, use_dc_delta)
}

/// Encode RGB image using YCbCr 4:2:0 + Huffman (version 4).
/// This is the recommended path for RGB images — best compression ratio.
pub fn encode_rgb_ycbcr(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_YUV420_V8, width, height, quality);

    let (y, cb, cr) = colorspace::rgb_to_ycbcr420(image, width, height);
    let cw = (width  + 1) / 2;
    let ch = (height + 1) / 2;

    let luma_table   = quant_table_for_quality_perceptual_v4(quality);
    let chroma_table = chroma_quant_table_for_quality_perceptual_v4(quality);
    let luma_sparsify = build_sparsify_thresholds(quality, false);
    let chroma_sparsify = build_sparsify_thresholds(quality, true);

    // Encode Y, Cb, Cr in parallel — each into its own buffer
    let blockizer_full   = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);

    let (y_buf, cb_buf, cr_buf) = if should_parallel_planes(width, height) {
        let (y_buf, (cb_buf, cr_buf)) = rayon::join(
            || {
                let mut blocks = blockizer_full.generate_blocks(&y);
                encode_channel_huffman(&mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify))
            },
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cb);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cr);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                )
            },
        );
        (y_buf, cb_buf, cr_buf)
    } else {
        let mut yb = blockizer_full.generate_blocks(&y);
        let y_buf = encode_channel_huffman(&mut yb, &luma_table, width, height, false, false, true, Some(&luma_sparsify));
        let mut cbb = blockizer_chroma.generate_blocks(&cb);
        let cb_buf = encode_channel_huffman(&mut cbb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        let mut crb = blockizer_chroma.generate_blocks(&cr);
        let cr_buf = encode_channel_huffman(&mut crb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        (y_buf, cb_buf, cr_buf)
    };

    bitstream::write_bytes(out, pos, &y_buf);
    bitstream::write_bytes(out, pos, &cb_buf);
    bitstream::write_bytes(out, pos, &cr_buf);
    write_icc_trailer(out, pos, icc);
}

/// Encode RGBA image using YCbCr 4:2:0 + Huffman + full-res alpha (version 5).
pub fn encode_rgba_ycbcr(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_YUV420A_V8, width, height, quality);

    let (y, cb, cr, a) = colorspace::rgba_to_ycbcr420a(image, width, height);
    let cw = (width  + 1) / 2;
    let ch = (height + 1) / 2;

    let luma_table   = quant_table_for_quality_perceptual_v4(quality);
    let chroma_table = chroma_quant_table_for_quality_perceptual_v4(quality);
    let luma_sparsify = build_sparsify_thresholds(quality, false);
    let chroma_sparsify = build_sparsify_thresholds(quality, true);

    let blockizer_full   = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);

    let (y_buf, cb_buf, cr_buf, a_buf) = if should_parallel_planes(width, height) {
        let ((y_buf, a_buf), (cb_buf, cr_buf)) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_full.generate_blocks(&y);
                        encode_channel_huffman(&mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify))
                    },
                    || {
                        let mut blocks = blockizer_full.generate_blocks(&a);
                        encode_channel_huffman(&mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify))
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cb);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cr);
                        encode_channel_huffman(&mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify))
                    },
                )
            },
        );
        (y_buf, cb_buf, cr_buf, a_buf)
    } else {
        let mut yb = blockizer_full.generate_blocks(&y);
        let y_buf = encode_channel_huffman(&mut yb, &luma_table, width, height, false, false, true, Some(&luma_sparsify));
        let mut cbb = blockizer_chroma.generate_blocks(&cb);
        let cb_buf = encode_channel_huffman(&mut cbb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        let mut crb = blockizer_chroma.generate_blocks(&cr);
        let cr_buf = encode_channel_huffman(&mut crb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify));
        let mut ab = blockizer_full.generate_blocks(&a);
        let a_buf = encode_channel_huffman(&mut ab, &luma_table, width, height, false, false, true, Some(&luma_sparsify));
        (y_buf, cb_buf, cr_buf, a_buf)
    };

    bitstream::write_bytes(out, pos, &y_buf);
    bitstream::write_bytes(out, pos, &cb_buf);
    bitstream::write_bytes(out, pos, &cr_buf);
    bitstream::write_bytes(out, pos, &a_buf);
    write_icc_trailer(out, pos, icc);
}

// ---------------------------------------------------------------------------
// Legacy RGB/RGBA RLE paths (v2/v3) — kept for backward compat
// ---------------------------------------------------------------------------

pub fn encode_rgb(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    // Default to the better YCbCr path; callers that need legacy RLE use encode_rgb_rle
    encode_rgb_ycbcr(image, width, height, quality, out, pos, icc);
}

pub fn encode_rgba(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    encode_rgba_ycbcr(image, width, height, quality, out, pos, icc);
}

/// Legacy RLE RGB encoder (v2). Used when explicit backward-compat is needed.
pub fn encode_rgb_rle(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_RGB, width, height, quality);
    let table = quant_table_for_quality(quality);
    let blockizer = Blockizer::new(width, height);
    let channel_bufs: Vec<Vec<u8>> = if should_parallel_planes(width, height) {
        (0..3usize).into_par_iter()
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgb(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    } else {
        (0..3usize)
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgb(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    };
    for buf in &channel_bufs { bitstream::write_bytes(out, pos, buf); }
    write_icc_trailer(out, pos, icc);
}

/// Legacy RLE RGBA encoder (v3).
pub fn encode_rgba_rle(
    image: &[u8], width: usize, height: usize, quality: u8,
    out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>,
) {
    write_header(out, pos, BG_MAGIC_RGBA, width, height, quality);
    let table = quant_table_for_quality(quality);
    let blockizer = Blockizer::new(width, height);
    let channel_bufs: Vec<Vec<u8>> = if should_parallel_planes(width, height) {
        (0..4usize).into_par_iter()
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgba(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    } else {
        (0..4usize)
            .map(|c| {
                let mut blocks = blockizer.generate_blocks_rgba(image, c);
                encode_channel_rle(&mut blocks, &table, width, height)
            })
            .collect()
    };
    for buf in &channel_bufs { bitstream::write_bytes(out, pos, buf); }
    write_icc_trailer(out, pos, icc);
}


// ===========================================================================
// Perceptual-weighted quantization (versions 20/21)
// ===========================================================================
//
// Motivation
// ----------
// The uniform quantizer treats all 64 DCT frequency bins the same way.  The
// human contrast-sensitivity function (CSF), however, falls off rapidly with
// spatial frequency: high-frequency coefficients carry detail the eye barely
// perceives, so spending entropy on them is wasteful.  This section adds an
// *optional* perceptually weighted quantizer: every frequency bin is scaled by
// a caller-supplied CSF weight before quantization.
//
// One table, both ends
// --------------------
// The same `[f32; 64]` table drives both sides:
//   * encode: effective step  s[i] = round(uniform_step[i] * w[zigzag[i]])
//   * decode: the inverse multiplier is rebuilt with the identical formula, so
//             the encoder and decoder can never drift apart.
//
// Weights are indexed in JPEG **zigzag order** (index 0 = DC, 63 = highest
// frequency) — the natural "low -> high frequency" ordering, which is what a
// monotone CSF table requires.  Quant tables themselves stay in raster order,
// matching the existing `quantize_block`/`dequantize_block` FFI kernels.
//
// Error policy
// ------------
// An empty table, a table of the wrong length, a non-finite/non-positive
// entry, or a table that is not monotone non-decreasing in zigzag order is
// rejected up front with [`PerceptualError`].  A decoder that encounters a
// stored table violating the same rules rejects the stream (`decode` returns
// `Err`); the legacy [`crate::decoder::decode`] entry point is untouched and
// simply rejects unknown version bytes as before.
//
// Wire format (v20 = RGB, v21 = RGBA).  Everything above the trailer layer is
// identical to v18/v19 (same YCbCr 4:2:0 planes, same Huffman profile, same
// sparsify/clamp pipeline); only the quant tables differ.
//
//   header  : "BG" + version(20|21) + width(u32 LE) + height(u32 LE) + quality
//   planes  : [len:u32][Huffman bits] for Y, Cb, Cr (, A for v21)
//   trailer?: "BGw" + flags:u8 + 64 * u16 LE zigzag percent (only if custom)
//   trailer?: "BGx" ICC profile (unchanged, written last)
//
// The default CSF table is *not* stored — both ends reconstruct it from the
// shared [`default_perceptual_weights`] function, costing 0 stream bytes.

/// Stream version byte for YCbCr 4:2:0 RGB with caller/CSF perceptual weights.
pub const BG_VERSION_PERCEPTUAL_WEIGHT_RGB: u8 = 20;
/// Stream version byte for the RGBA (full-res alpha) variant.
pub const BG_VERSION_PERCEPTUAL_WEIGHT_RGBA: u8 = 21;

const BG_MAGIC_YUV420_V20: &[u8; 3] = b"BG\x14"; // 0x14 = 20
const BG_MAGIC_YUV420A_V21: &[u8; 3] = b"BG\x15"; // 0x15 = 21

/// Number of frequency bins in one 8×8 block / one weight table.
pub const PERCEPTUAL_WEIGHT_LEN: usize = 64;
/// "BGw" custom-weight trailer: 3 magic + 1 flags + 64 * u16.
pub const CUSTOM_WEIGHT_TRAILER_LEN: usize = 3 + 1 + PERCEPTUAL_WEIGHT_LEN * 2;
const CUSTOM_WEIGHT_FLAGS: u8 = 1;
/// Fixed-point percent used for the integer-only shared table computation.
const PERCENT_SCALE: i32 = 100;
/// Largest weight the API accepts (keeps effective steps inside i16 range).
const MAX_WEIGHT: f32 = 32.0;

/// Failures reported by the perceptual-weight API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerceptualError {
    /// Slice was empty or did not contain exactly 64 entries.
    InvalidLength,
    /// An entry was NaN/infinite/zero/negative — a usable weight must be a
    /// finite, strictly positive number.
    InvalidValue,
    /// Zigzag-ordered weights must never decrease (DC → highest frequency).
    NonMonotonic,
    /// Stream buffer was truncated or carried an unsupported layout.
    MalformedStream,
}

impl core::fmt::Display for PerceptualError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let msg = match self {
            PerceptualError::InvalidLength => "perceptual weight table must contain exactly 64 entries",
            PerceptualError::InvalidValue => "perceptual weights must be finite and strictly positive",
            PerceptualError::NonMonotonic => "perceptual weight table must be non-decreasing in zigzag order",
            PerceptualError::MalformedStream => "malformed perceptual-weight .bg stream",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for PerceptualError {}

/// Default contrast-sensitivity weights, zigzag order, monotone non-decreasing.
///
/// Model: `w(z) = 1 + G * (z/63)^p`, with `G = 2.0`, `p = 1.5`.  The DC bin and
/// the lowest AC bins keep weight 1.0 (no extra loss where the eye is most
/// sensitive); the highest bin reaches 3.0, i.e. its effective quant step can
/// be up to 3× coarser.  Rounded to two decimals, then made strictly monotone
/// (cumulative max) so the result always passes validation.  Empirically
/// (see `four_material_evaluation_report`) this leaves smooth/textured
/// payloads untouched while shrinking noise ~8% and text ~10% at q85.
pub fn default_perceptual_weights() -> [f32; PERCEPTUAL_WEIGHT_LEN] {
    const G: f32 = 2.0;
    const P: f32 = 1.5;
    let mut out = [1.0f32; PERCEPTUAL_WEIGHT_LEN];
    let mut running = 0.0f32;
    for (z, slot) in out.iter_mut().enumerate() {
        let t = z as f32 / (PERCEPTUAL_WEIGHT_LEN as f32 - 1.0);
        let w = 1.0 + G * t.powf(P);
        let rounded = (w * 100.0).round() / 100.0;
        running = running.max(rounded);
        *slot = running;
    }
    out
}

/// Validate a caller-supplied weight table: 64 finite positive entries that
/// never decrease in zigzag (frequency) order.
pub fn validate_perceptual_weights(weights: &[f32]) -> Result<(), PerceptualError> {
    if weights.len() != PERCEPTUAL_WEIGHT_LEN {
        return Err(PerceptualError::InvalidLength);
    }
    let mut prev = f32::MIN;
    for &w in weights {
        if !w.is_finite() || w <= 0.0 || w > MAX_WEIGHT {
            return Err(PerceptualError::InvalidValue);
        }
        if w < prev {
            return Err(PerceptualError::NonMonotonic);
        }
        prev = w;
    }
    Ok(())
}

/// Same checks for a decoded fixed-point (zigzag percent) table.
fn validate_perceptual_percents(percents: &[u16]) -> Result<(), PerceptualError> {
    if percents.len() != PERCEPTUAL_WEIGHT_LEN {
        return Err(PerceptualError::InvalidLength);
    }
    // 1% .. 3200% mirrors the accepted float range (0 < w <= MAX_WEIGHT).
    let min_percent = 1u16;
    let max_percent = (MAX_WEIGHT * 100.0) as u16;
    let mut prev: u16 = 0;
    for &p in percents {
        if p < min_percent || p > max_percent {
            return Err(PerceptualError::InvalidValue);
        }
        if p < prev {
            return Err(PerceptualError::NonMonotonic);
        }
        prev = p;
    }
    Ok(())
}

/// Convert float weights to the shared fixed-point representation (zigzag
/// percent).  Both encoder and decoder use these integers, so the round trip
/// never depends on host float behavior.
fn weights_to_percents(weights: &[f32]) -> Result<[u16; PERCEPTUAL_WEIGHT_LEN], PerceptualError> {
    validate_perceptual_weights(weights)?;
    let mut out = [0u16; PERCEPTUAL_WEIGHT_LEN];
    for (i, &w) in weights.iter().enumerate() {
        out[i] = (w * PERCENT_SCALE as f32).round().clamp(1.0, u16::MAX as f32) as u16;
    }
    // Re-check the integerized table: rounding must not break monotonicity.
    let mut prev: u16 = 0;
    for &p in out.iter() {
        if p < prev {
            return Err(PerceptualError::NonMonotonic);
        }
        prev = p;
    }
    Ok(out)
}

/// Apply zigzag-ordered percents to a raster-order quant table.  This is the
/// single formula used by both encoder and decoder:
///
/// ```text
/// step[raster] = clamp(uniform_step[raster] * percent[zigzag_index] / 100, 1..255)
/// ```
///
/// `percent[zig]` is indexed by zigzag position `zig`; `ZIGZAG[zig]` gives the
/// raster bin that position maps to.
fn apply_weights_to_table(base: &[i16; 64], zigzag_percent: &[u16; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    for zig in 0..64 {
        let raster = ZIGZAG[zig];
        let v = base[raster] as i32 * zigzag_percent[zig] as i32 / PERCENT_SCALE;
        out[raster] = v.clamp(1, 255) as i16;
    }
    out
}

/// Build the luma/chroma effective quant tables for given quality + weights.
/// Returns the raster tables plus the fixed-point table shared with decoding.
fn build_weighted_quant_tables(
    quality: u8,
    weights: &[f32],
) -> Result<([i16; 64], [i16; 64], [u16; 64]), PerceptualError> {
    let percents = weights_to_percents(weights)?;
    let luma_base = quant_table_for_quality_perceptual_v4(quality);
    let chroma_base = chroma_quant_table_for_quality_perceptual_v4(quality);
    let luma = apply_weights_to_table(&luma_base, &percents);
    let chroma = apply_weights_to_table(&chroma_base, &percents);
    Ok((luma, chroma, percents))
}

fn write_weight_trailer(out: &mut [u8], pos: &mut i32, percents: &[u16; 64]) {
    bitstream::write_bytes(out, pos, b"BGw");
    bitstream::write_byte(out, pos, CUSTOM_WEIGHT_FLAGS);
    for &p in percents {
        for b in p.to_le_bytes() {
            bitstream::write_byte(out, pos, b);
        }
    }
}

/// Parse the optional "BGw" trailer starting at `pos`.  Returns the decoded
/// zigzag percents (or `None` for the built-in default table) and the byte
/// position just past the trailer.
fn parse_weight_trailer(buffer: &[u8], pos: usize) -> Result<(Option<[u16; 64]>, usize), PerceptualError> {
    if pos + 3 > buffer.len() || &buffer[pos..pos + 3] != b"BGw" {
        return Ok((None, pos));
    }
    let body = pos + 3;
    if body + 1 + PERCEPTUAL_WEIGHT_LEN * 2 > buffer.len() {
        return Err(PerceptualError::MalformedStream);
    }
    if buffer[body] != CUSTOM_WEIGHT_FLAGS {
        return Err(PerceptualError::MalformedStream);
    }
    let mut percents = [0u16; PERCEPTUAL_WEIGHT_LEN];
    let mut p = body + 1;
    for slot in percents.iter_mut() {
        *slot = u16::from_le_bytes([buffer[p], buffer[p + 1]]);
        p += 2;
    }
    validate_perceptual_percents(&percents)?;
    Ok((Some(percents), p))
}

/// Encode RGB to a v20 perceptual-weight .bg stream.
///
/// `weights` is a 64-entry zigzag-ordered CSF table; pass
/// [`default_perceptual_weights`] for the built-in contrast-sensitivity curve.
/// The uniform quant path (v1–v19) is unaffected.
pub fn encode_rgb_weighted(
    image: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    weights: &[f32],
    out: &mut [u8],
    pos: &mut i32,
    icc: Option<&[u8]>,
) -> Result<(), PerceptualError> {
    // Normalize like the legacy header: 0 means default 50, then clamp.
    let quality = if quality == 0 { 50 } else { quality.clamp(1, 100) };
    let (luma_table, chroma_table, percents) = build_weighted_quant_tables(quality, weights)?;
    let default_percents = weights_to_percents(&default_perceptual_weights())?;
    let is_custom = percents != default_percents;

    write_header(out, pos, BG_MAGIC_YUV420_V20, width, height, quality);
    let (y, cb, cr) = colorspace::rgb_to_ycbcr420(image, width, height);
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let luma_sparsify = build_sparsify_thresholds(quality, false);
    let chroma_sparsify = build_sparsify_thresholds(quality, true);

    let blockizer_full = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);

    let (y_buf, cb_buf, cr_buf) = if should_parallel_planes(width, height) {
        let (y_buf, (cb_buf, cr_buf)) = rayon::join(
            || {
                let mut blocks = blockizer_full.generate_blocks(&y);
                encode_channel_huffman(
                    &mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify),
                )
            },
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cb);
                        encode_channel_huffman(
                            &mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
                        )
                    },
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cr);
                        encode_channel_huffman(
                            &mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
                        )
                    },
                )
            },
        );
        (y_buf, cb_buf, cr_buf)
    } else {
        let mut yb = blockizer_full.generate_blocks(&y);
        let y_buf = encode_channel_huffman(
            &mut yb, &luma_table, width, height, false, false, true, Some(&luma_sparsify),
        );
        let mut cbb = blockizer_chroma.generate_blocks(&cb);
        let cb_buf = encode_channel_huffman(
            &mut cbb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
        );
        let mut crb = blockizer_chroma.generate_blocks(&cr);
        let cr_buf = encode_channel_huffman(
            &mut crb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
        );
        (y_buf, cb_buf, cr_buf)
    };

    bitstream::write_bytes(out, pos, &y_buf);
    bitstream::write_bytes(out, pos, &cb_buf);
    bitstream::write_bytes(out, pos, &cr_buf);
    if is_custom {
        write_weight_trailer(out, pos, &percents);
    }
    write_icc_trailer(out, pos, icc);
    Ok(())
}

/// Encode RGBA to a v21 perceptual-weight .bg stream (full-res alpha plane).
#[allow(clippy::too_many_arguments)]
pub fn encode_rgba_weighted(
    image: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    weights: &[f32],
    out: &mut [u8],
    pos: &mut i32,
    icc: Option<&[u8]>,
) -> Result<(), PerceptualError> {
    // Normalize like the legacy header: 0 means default 50, then clamp.
    let quality = if quality == 0 { 50 } else { quality.clamp(1, 100) };
    let (luma_table, chroma_table, percents) = build_weighted_quant_tables(quality, weights)?;
    let default_percents = weights_to_percents(&default_perceptual_weights())?;
    let is_custom = percents != default_percents;

    write_header(out, pos, BG_MAGIC_YUV420A_V21, width, height, quality);
    let (y, cb, cr, a) = colorspace::rgba_to_ycbcr420a(image, width, height);
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let luma_sparsify = build_sparsify_thresholds(quality, false);
    let chroma_sparsify = build_sparsify_thresholds(quality, true);

    let blockizer_full = Blockizer::new(width, height);
    let blockizer_chroma = Blockizer::new(cw, ch);

    let (y_buf, cb_buf, cr_buf, a_buf) = if should_parallel_planes(width, height) {
        let ((y_buf, a_buf), (cb_buf, cr_buf)) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_full.generate_blocks(&y);
                        encode_channel_huffman(
                            &mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify),
                        )
                    },
                    || {
                        let mut blocks = blockizer_full.generate_blocks(&a);
                        encode_channel_huffman(
                            &mut blocks, &luma_table, width, height, false, false, true, Some(&luma_sparsify),
                        )
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cb);
                        encode_channel_huffman(
                            &mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
                        )
                    },
                    || {
                        let mut blocks = blockizer_chroma.generate_blocks(&cr);
                        encode_channel_huffman(
                            &mut blocks, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
                        )
                    },
                )
            },
        );
        (y_buf, cb_buf, cr_buf, a_buf)
    } else {
        let mut yb = blockizer_full.generate_blocks(&y);
        let y_buf = encode_channel_huffman(
            &mut yb, &luma_table, width, height, false, false, true, Some(&luma_sparsify),
        );
        let mut cbb = blockizer_chroma.generate_blocks(&cb);
        let cb_buf = encode_channel_huffman(
            &mut cbb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
        );
        let mut crb = blockizer_chroma.generate_blocks(&cr);
        let cr_buf = encode_channel_huffman(
            &mut crb, &chroma_table, cw, ch, true, true, true, Some(&chroma_sparsify),
        );
        let mut ab = blockizer_full.generate_blocks(&a);
        let a_buf = encode_channel_huffman(
            &mut ab, &luma_table, width, height, false, false, true, Some(&luma_sparsify),
        );
        (y_buf, cb_buf, cr_buf, a_buf)
    };

    bitstream::write_bytes(out, pos, &y_buf);
    bitstream::write_bytes(out, pos, &cb_buf);
    bitstream::write_bytes(out, pos, &cr_buf);
    bitstream::write_bytes(out, pos, &a_buf);
    if is_custom {
        write_weight_trailer(out, pos, &percents);
    }
    write_icc_trailer(out, pos, icc);
    Ok(())
}

/// Decode a v20/v21 perceptual-weight stream.  The weight table is taken from
/// the stream's "BGw" trailer when present, otherwise rebuilt with the shared
/// [`default_perceptual_weights`] — guaranteeing encoder/decoder use one table.
pub fn decode_weighted(
    buffer: &[u8],
    out_pixels: &mut [u8],
    out_width: &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
) -> Result<(), PerceptualError> {
    const HEADER: usize = 3 + 4 + 4 + 1;
    if buffer.len() < HEADER || &buffer[0..2] != b"BG" {
        return Err(PerceptualError::MalformedStream);
    }
    let version = buffer[2];
    if version != BG_VERSION_PERCEPTUAL_WEIGHT_RGB && version != BG_VERSION_PERCEPTUAL_WEIGHT_RGBA {
        return Err(PerceptualError::MalformedStream);
    }
    let channels = if version == BG_VERSION_PERCEPTUAL_WEIGHT_RGB { 3u32 } else { 4u32 };
    let width = u32::from_le_bytes(buffer[3..7].try_into().unwrap()) as usize;
    let height = u32::from_le_bytes(buffer[7..11].try_into().unwrap()) as usize;
    let quality = if buffer[11] == 0 { 50 } else { buffer[11] };
    if width == 0 || height == 0 || width > 65536 || height > 65536 {
        return Err(PerceptualError::MalformedStream);
    }
    if out_pixels.len() < width * height * channels as usize {
        return Err(PerceptualError::MalformedStream);
    }

    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let n_full = ((width + 7) / 8) * ((height + 7) / 8);
    let n_chroma = ((cw + 7) / 8) * ((ch + 7) / 8);
    // ---- planes (identical Huffman profile to v18/v19) ----
    let mut pos = HEADER;
    let (mut y_blocks, p1) =
        huffman::decode_plane_with_profile(buffer, pos, n_full, false, false, true)
            .ok_or(PerceptualError::MalformedStream)?;
    pos = p1;
    let (mut cb_blocks, p2) =
        huffman::decode_plane_with_profile(buffer, pos, n_chroma, true, true, true)
            .ok_or(PerceptualError::MalformedStream)?;
    pos = p2;
    let (mut cr_blocks, p3) =
        huffman::decode_plane_with_profile(buffer, pos, n_chroma, true, true, true)
            .ok_or(PerceptualError::MalformedStream)?;
    pos = p3;
    let mut a_blocks: Vec<Block> = if channels == 4 {
        let (blocks, p4) = huffman::decode_plane_with_profile(buffer, pos, n_full, false, false, true)
            .ok_or(PerceptualError::MalformedStream)?;
        pos = p4;
        blocks
    } else {
        Vec::new()
    };

    // ---- resolve the shared weight table ----
    let (stored, after_weights) = parse_weight_trailer(buffer, pos)?;
    let percents: [u16; 64] = match stored {
        Some(p) => p,
        None => weights_to_percents(&default_perceptual_weights())?,
    };
    let _ = after_weights; // any remaining bytes are the optional ICC trailer
    let luma_q = apply_weights_to_table(&quant_table_for_quality_perceptual_v4(quality), &percents);
    let chroma_q = apply_weights_to_table(&chroma_quant_table_for_quality_perceptual_v4(quality), &percents);

    // ---- dequant + IDCT (same kernels as every other version) ----
    let dequant_plane = |blocks: &mut [Block], table: &[i16; 64]| {
        if should_parallel_dequant(blocks.len(), width, height) {
            blocks.par_chunks_mut(BLOCK_TILE_SIZE).for_each(|chunk| {
                for block in chunk.iter_mut() {
                    dequantize(block, table);
                    dct::idct(block);
                }
            });
        } else {
            for block in blocks.iter_mut() {
                dequantize(block, table);
                dct::idct(block);
            }
        }
    };
    dequant_plane(&mut y_blocks, &luma_q);
    dequant_plane(&mut cb_blocks, &chroma_q);
    dequant_plane(&mut cr_blocks, &chroma_q);
    if channels == 4 {
        dequant_plane(&mut a_blocks, &luma_q);
    }

    let mut y_plane = vec![0u8; width * height];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];
    let bw = (width + 7) / 8;
    let bw_c = (cw + 7) / 8;
    for (idx, block) in y_blocks.iter().enumerate() {
        write_block_to_plane(block, &mut y_plane, width, height, (idx % bw) * 8, (idx / bw) * 8);
    }
    for (idx, block) in cb_blocks.iter().enumerate() {
        write_block_to_plane(block, &mut cb_plane, cw, ch, (idx % bw_c) * 8, (idx / bw_c) * 8);
    }
    for (idx, block) in cr_blocks.iter().enumerate() {
        write_block_to_plane(block, &mut cr_plane, cw, ch, (idx % bw_c) * 8, (idx / bw_c) * 8);
    }

    if channels == 3 {
        colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, width, height, out_pixels);
    } else {
        let mut a_plane = vec![0u8; width * height];
        for (idx, block) in a_blocks.iter().enumerate() {
            write_block_to_plane(block, &mut a_plane, width, height, (idx % bw) * 8, (idx / bw) * 8);
        }
        colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, width, height, out_pixels);
    }

    *out_width = width as u32;
    *out_height = height as u32;
    *out_channels = channels;
    Ok(())
}

// ===========================================================================
// Tests for perceptual-weighted quantization and legacy-path preservation.
// In test builds the C FFI is not linked, so the four C symbols are shimmed
// here with pure-Rust equivalents matching c/quant.c / dct.rs semantics.
// ===========================================================================
#[cfg(test)]
mod perceptual_tests {
    use super::*;
    use crate::block::Block;
    use crate::dct::{dct_reference, idct_reference};
    use std::time::Instant;

    #[no_mangle]
    pub extern "C" fn quantize_block(block: *mut i16, table: *const i16) {
        for i in 0..64 {
            unsafe {
                let b = *block.add(i);
                let t = *table.add(i);
                *block.add(i) = ((b as f32) / (t as f32)).round_ties_even() as i16;
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn dequantize_block(block: *mut i16, table: *const i16) {
        for i in 0..64 {
            unsafe {
                let v = *block.add(i) as i32 * *table.add(i) as i32;
                *block.add(i) = v.clamp(-32768, 32767) as i16;
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn bitgrain_dct_block(block: *mut i16) {
        let block = block as *mut [i16; 64];
        unsafe {
            let blk = Block { data: std::ptr::read(block) };
            std::ptr::write(block, dct_reference(&blk));
        }
    }

    #[no_mangle]
    pub extern "C" fn bitgrain_idct_block(block: *mut i16) {
        let block = block as *mut [i16; 64];
        unsafe {
            let data = std::ptr::read(block);
            std::ptr::write(block, idct_reference(&data));
        }
    }

    // ---- golden v18 blob produced by the pre-change encoder (18x13 RGB q85) ----
    const GOLDEN_V18: [u8; 148] = [
        066,071,018,018,000,000,000,013,000,000,000,085,115,000,000,000,169,253,155,027,
        067,231,127,102,249,118,178,196,219,210,054,219,252,127,223,255,000,087,079,109,
        018,087,086,251,087,150,222,103,150,158,099,071,181,029,255,000,244,093,084,159,
        076,104,229,104,229,211,181,151,117,111,153,162,211,100,217,093,098,104,074,154,
        012,186,180,055,082,043,091,237,143,251,174,251,234,132,086,016,193,167,091,223,
        121,113,164,115,237,182,085,133,054,055,220,174,186,047,014,106,243,070,178,218,
        106,058,100,048,063,205,028,109,165,035,087,006,000,000,000,175,236,067,063,236,
        131,003,000,000,000,126,208,131,
    ];

    fn sample_rgb(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut img = vec![0u8; w * h * 3];
        let mut s = seed;
        for (i, px) in img.chunks_exact_mut(3).enumerate() {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            px[0] = ((s >> 16) & 0xff) as u8;
            px[1] = ((i * 37 + 11) & 0xff) as u8;
            px[2] = (((i / w) * 19) & 0xff) as u8;
        }
        img
    }

    fn encode_uniform(img: &[u8], w: usize, h: usize, q: u8) -> Vec<u8> {
        let mut out = vec![0u8; 1 << 20];
        let mut pos = 0i32;
        super::super::encoder::encode_rgb(img, w, h, q, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        out
    }

    fn encode_weighted(img: &[u8], w: usize, h: usize, q: u8, weights: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; 1 << 20];
        let mut pos = 0i32;
        encode_rgb_weighted(img, w, h, q, weights, &mut out, &mut pos, None).unwrap();
        out.truncate(pos as usize);
        out
    }

    fn decode_to_rgb(buf: &[u8], w: usize, h: usize) -> Vec<u8> {
        let mut pixels = vec![0u8; w * h * 3];
        let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
        if buf[2] == BG_VERSION_PERCEPTUAL_WEIGHT_RGB || buf[2] == BG_VERSION_PERCEPTUAL_WEIGHT_RGBA {
            decode_weighted(buf, &mut pixels, &mut ow, &mut oh, &mut oc).unwrap();
        } else {
            assert!(crate::decoder::decode(
                buf, &mut pixels, &mut ow, &mut oh, &mut oc, None
            ));
        }
        assert_eq!((ow as usize, oh as usize, oc), (w, h, 3));
        pixels
    }

    // PSNR identical in formula to c/metrics.c::bitgrain_psnr.
    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let mut sse = 0u64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = *x as i32 - *y as i32;
            sse += (d * d) as u64;
        }
        if sse == 0 {
            return 99.0;
        }
        let mse = sse as f64 / a.len() as f64;
        10.0 * (65025.0 / mse).log10()
    }

    // Global-window SSIM identical in formula to c/metrics.c::bitgrain_ssim.
    fn ssim(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len() as f64;
        let (c1, c2): (f64, f64) = (6.5025, 58.5225);
        let mu_x = a.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mu_y = b.iter().map(|&v| v as f64).sum::<f64>() / n;
        let (mut sx2, mut sy2, mut sxy) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b.iter()) {
            let dx = *x as f64 - mu_x;
            let dy = *y as f64 - mu_y;
            sx2 += dx * dx;
            sy2 += dy * dy;
            sxy += dx * dy;
        }
        sx2 /= n;
        sy2 /= n;
        sxy /= n;
        let l = (2.0 * mu_x * mu_y + c1) / (mu_x * mu_x + mu_y * mu_y + c1);
        let c = (2.0 * sx2.sqrt() * sy2.sqrt() + c2) / (sx2 + sy2 + c2);
        let s = (sxy + c2 / 2.0) / (sx2.sqrt() * sy2.sqrt() + c2 / 2.0);
        l * c * s
    }

    // -------------------------------------------------------------------
    // Weight table validation
    // -------------------------------------------------------------------

    #[test]
    fn empty_table_is_rejected() {
        let img = sample_rgb(8, 8, 1);
        let mut out = vec![0u8; 1 << 16];
        let mut pos = 0;
        let empty: [f32; 0] = [];
        let err = encode_rgb_weighted(&img, 8, 8, 85, &empty, &mut out, &mut pos, None)
            .unwrap_err();
        assert_eq!(err, PerceptualError::InvalidLength);
        // Slice of the wrong length is rejected identically.
        let short = [1.0f32; 63];
        let err = encode_rgb_weighted(&img, 8, 8, 85, &short, &mut out, &mut pos, None)
            .unwrap_err();
        assert_eq!(err, PerceptualError::InvalidLength);
    }

    #[test]
    fn non_monotonic_table_is_rejected() {
        let img = sample_rgb(8, 8, 2);
        let mut bad = default_perceptual_weights();
        bad[10] = bad[9] - 0.5; // zigzag-order decrease
        let mut out = vec![0u8; 1 << 16];
        let mut pos = 0;
        let err = encode_rgb_weighted(&img, 8, 8, 85, &bad, &mut out, &mut pos, None)
            .unwrap_err();
        assert_eq!(err, PerceptualError::NonMonotonic);
    }

    #[test]
    fn invalid_value_tables_are_rejected() {
        let img = sample_rgb(8, 8, 3);
        let mut out = vec![0u8; 1 << 16];
        let mut pos = 0;
        let mut nan = default_perceptual_weights();
        nan[5] = f32::NAN;
        assert_eq!(
            encode_rgb_weighted(&img, 8, 8, 85, &nan, &mut out, &mut pos, None).unwrap_err(),
            PerceptualError::InvalidValue
        );
        let mut zero = default_perceptual_weights();
        zero[0] = 0.0;
        assert_eq!(
            encode_rgb_weighted(&img, 8, 8, 85, &zero, &mut out, &mut pos, None).unwrap_err(),
            PerceptualError::InvalidValue
        );
        let mut neg = default_perceptual_weights();
        neg[0] = -1.0;
        assert_eq!(
            encode_rgb_weighted(&img, 8, 8, 85, &neg, &mut out, &mut pos, None).unwrap_err(),
            PerceptualError::InvalidValue
        );
        let mut huge = default_perceptual_weights();
        huge[63] = 100.0;
        assert_eq!(
            encode_rgb_weighted(&img, 8, 8, 85, &huge, &mut out, &mut pos, None).unwrap_err(),
            PerceptualError::InvalidValue
        );
    }

    #[test]
    fn default_table_is_valid_and_monotone() {
        let w = default_perceptual_weights();
        assert_eq!(w.len(), 64);
        validate_perceptual_weights(&w).unwrap();
        for pair in w.windows(2) {
            assert!(pair[1] >= pair[0]);
        }
        assert!((w[0] - 1.0).abs() < 1e-6);
        assert!(w[63] > 2.0);
    }

    // -------------------------------------------------------------------
    // Legacy uniform paths must remain byte-for-byte unchanged
    // -------------------------------------------------------------------

    #[test]
    fn uniform_v18_path_is_byte_identical_to_baseline() {
        let img = sample_rgb(18, 13, 12_345);
        let blob = encode_uniform(&img, 18, 13, 85);
        assert_eq!(blob, GOLDEN_V18, "uniform v18 encoder output changed");
        assert_eq!(blob[2], 18);
    }

    #[test]
    fn uniform_legacy_tables_are_unchanged() {
        // q85 -> JPEG scale factor 30; the uniform table must match the
        // shipped formula byte-for-byte (this is the path the task protects).
        let t = quant_table_for_quality(85);
        assert_eq!(t[0], (16 * 30 + 50) / 100); // 5
        assert_eq!(t[63], (99 * 30 + 50) / 100); // 30
        // The pre-existing perceptual v4 profile is deterministic and keeps
        // the DC bin at the uniform step.
        let p = quant_table_for_quality_perceptual_v4(85);
        assert_eq!(p[0], t[0]);
        assert_eq!(p, quant_table_for_quality_perceptual_v4(85));
        // High frequencies are coarser than uniform in the v4 profile.
        assert!(p[63] >= t[63]);
    }

    #[test]
    fn legacy_version_bytes_are_untouched() {
        let img = sample_rgb(16, 16, 7);
        let rgb = encode_uniform(&img, 16, 16, 80);
        assert_eq!(&rgb[0..3], b"BG\x12"); // v18 RGB
        let mut rgba = vec![0u8; 16 * 16 * 4];
        for (i, v) in rgba.iter_mut().enumerate() {
            *v = (i.wrapping_mul(13) & 0xff) as u8;
        }
        let mut out = vec![0u8; 1 << 20];
        let mut pos = 0;
        super::super::encoder::encode_rgba(&rgba, 16, 16, 80, &mut out, &mut pos, None);
        out.truncate(pos as usize);
        assert_eq!(&out[0..3], b"BG\x13"); // v19 RGBA
    }

    // -------------------------------------------------------------------
    // Weighted v20/v21 behavior
    // -------------------------------------------------------------------

    #[test]
    fn weighted_roundtrip_rgb_default_table() {
        let (w, h) = (32, 24);
        let img = sample_rgb(w, h, 99);
        let blob = encode_weighted(&img, w, h, 85, &default_perceptual_weights());
        assert_eq!(&blob[0..3], b"BG\x14"); // v20
        // No "BGw" trailer for the built-in table: zero per-stream overhead.
        assert!(!blob.windows(3).any(|m| m == b"BGw"));
        let dec = decode_to_rgb(&blob, w, h);
        // Same image through the uniform path; weighted may trade a little PSNR
        // for size, but must stay within 2 dB of it.
        let dec_u = decode_to_rgb(&encode_uniform(&img, w, h, 85), w, h);
        let (q_w, q_u) = (psnr(&img, &dec), psnr(&img, &dec_u));
        assert!(q_w >= q_u - 2.0, "weighted PSNR {q_w} too far below uniform {q_u}");
        // Global SSIM on pure LCG noise is statistically jumpy at 32x24
        // (the 64x64 four-material report shows deltas < 0.01); allow 0.1.
        let (s_w, s_u) = (ssim(&img, &dec), ssim(&img, &dec_u));
        assert!(s_w >= s_u - 0.10, "weighted SSIM {s_w} vs uniform {s_u}");
    }

    #[test]
    fn weighted_roundtrip_rgb_custom_table_persists_weights() {
        let (w, h) = (32, 24);
        let img = sample_rgb(w, h, 4242);
        let mut custom = [1.0f32; 64];
        for (z, slot) in custom.iter_mut().enumerate() {
            *slot = 1.0 + 2.0 * ((z as f32 / 63.0).powi(3));
        }
        validate_perceptual_weights(&custom).unwrap();
        let blob = encode_weighted(&img, w, h, 85, &custom);
        assert_eq!(&blob[0..3], b"BG\x14");
        // Custom table must be embedded via the "BGw" trailer (132 bytes).
        let trailer = blob.len() - CUSTOM_WEIGHT_TRAILER_LEN;
        assert_eq!(&blob[trailer..trailer + 3], b"BGw");
        let dec = decode_to_rgb(&blob, w, h);
        let dec_u = decode_to_rgb(&encode_uniform(&img, w, h, 85), w, h);
        assert!(psnr(&img, &dec) >= psnr(&img, &dec_u) - 3.0);
    }

    #[test]
    fn weighted_encoding_is_deterministic() {
        let img = sample_rgb(24, 24, 55);
        let b1 = encode_weighted(&img, 24, 24, 80, &default_perceptual_weights());
        let b2 = encode_weighted(&img, 24, 24, 80, &default_perceptual_weights());
        assert_eq!(b1, b2);
    }

    #[test]
    fn weighted_roundtrip_rgba_v21() {
        let (w, h) = (20, 16);
        let mut img = vec![0u8; w * h * 4];
        for (i, px) in img.chunks_exact_mut(4).enumerate() {
            px[0] = (i * 7 & 0xff) as u8;
            px[1] = (i * 11 & 0xff) as u8;
            px[2] = (i * 13 & 0xff) as u8;
            px[3] = if (i / w) % 2 == 0 { 255 } else { 120 };
        }
        let mut out = vec![0u8; 1 << 20];
        let mut pos = 0;
        encode_rgba_weighted(
            &img, w, h, 85, &default_perceptual_weights(), &mut out, &mut pos, None,
        )
        .unwrap();
        out.truncate(pos as usize);
        assert_eq!(&out[0..3], b"BG\x15"); // v21
        let mut pixels = vec![0u8; w * h * 4];
        let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
        decode_weighted(&out, &mut pixels, &mut ow, &mut oh, &mut oc).unwrap();
        assert_eq!((ow as usize, oh as usize, oc), (w, h, 4));
        let mut ref_out = vec![0u8; 1 << 20];
        let mut rpos = 0;
        super::super::encoder::encode_rgba(&img, w, h, 85, &mut ref_out, &mut rpos, None);
        ref_out.truncate(rpos as usize);
        let mut ref_px = vec![0u8; w * h * 4];
        let (mut rw, mut rh, mut rc) = (0u32, 0u32, 0u32);
        assert!(crate::decoder::decode(
            &ref_out, &mut ref_px, &mut rw, &mut rh, &mut rc, None
        ));
        assert!(psnr(&img, &pixels) >= psnr(&img, &ref_px) - 3.0);
    }

    #[test]
    fn encoder_decoder_share_one_table_identical_steps() {
        // Rebuild the decoder-side luma table exactly as decode_weighted does
        // and require it to match the encoder-side effective table.
        let q = 85u8;
        let weights = default_perceptual_weights();
        let (enc_luma, _enc_chroma, percents) = build_weighted_quant_tables(q, &weights).unwrap();
        let rebuilt_percents = weights_to_percents(&weights).unwrap();
        assert_eq!(percents, rebuilt_percents);
        let dec_luma = apply_weights_to_table(
            &quant_table_for_quality_perceptual_v4(q),
            &rebuilt_percents,
        );
        assert_eq!(enc_luma, dec_luma);
    }

    #[test]
    fn weighted_does_not_inflate_size_and_keeps_dc_protection() {
        let (w, h) = (64, 64);
        let img = sample_rgb(w, h, 777);
        let uniform = encode_uniform(&img, w, h, 85);
        let weighted = encode_weighted(&img, w, h, 85, &default_perceptual_weights());
        // Default-table stream carries no weight overhead and must not grow.
        assert!(weighted.len() <= uniform.len());
        // DC (zigzag 0) weight is 1.0: the DC effective step is untouched.
        let percents = weights_to_percents(&default_perceptual_weights()).unwrap();
        assert_eq!(percents[0], 100);
        let (luma, _c, _p) = build_weighted_quant_tables(85, &default_perceptual_weights()).unwrap();
        assert_eq!(luma[0], quant_table_for_quality_perceptual_v4(85)[0]);
    }

    #[test]
    fn malformed_weighted_streams_are_rejected() {
        let img = sample_rgb(16, 16, 4);
        let good = encode_weighted(&img, 16, 16, 85, &default_perceptual_weights());
        let mut pixels = vec![0u8; 16 * 16 * 3];
        let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);

        // Truncated header.
        assert_eq!(
            decode_weighted(&good[..6], &mut pixels, &mut ow, &mut oh, &mut oc).unwrap_err(),
            PerceptualError::MalformedStream
        );
        // Wrong version byte routed here must be rejected.
        let mut wrong = good.clone();
        wrong[2] = 18;
        assert_eq!(
            decode_weighted(&wrong, &mut pixels, &mut ow, &mut oh, &mut oc).unwrap_err(),
            PerceptualError::MalformedStream
        );
        // Custom table stream with a broken monotonicity in the trailer.
        let mut custom = [1.0f32; 64];
        for (z, slot) in custom.iter_mut().enumerate() {
            *slot = 1.0 + 1.5 * (z as f32 / 63.0);
        }
        let mut with_trailer = encode_weighted(&img, 16, 16, 85, &custom);
        let t = with_trailer.len() - CUSTOM_WEIGHT_TRAILER_LEN;
        // Lower a high-frequency percent below its predecessor.
        let last_p = with_trailer.len() - 2;
        with_trailer[last_p] = 1;
        with_trailer[last_p + 1] = 0;
        assert_eq!(&with_trailer[t..t + 3], b"BGw");
        assert_eq!(
            decode_weighted(&with_trailer, &mut pixels, &mut ow, &mut oh, &mut oc).unwrap_err(),
            PerceptualError::NonMonotonic
        );
        // Truncated trailer.
        let cut = with_trailer.len() - 10;
        assert_eq!(
            decode_weighted(&with_trailer[..cut], &mut pixels, &mut ow, &mut oh, &mut oc).unwrap_err(),
            PerceptualError::MalformedStream
        );
    }

    // -------------------------------------------------------------------
    // Four-class material evaluation (printed via --nocapture)
    // -------------------------------------------------------------------

    fn make_material(kind: &str, w: usize, h: usize) -> Vec<u8> {
        let mut img = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let (r, g, b) = match kind {
                    // Smooth: low-amplitude bilinear gradient, no AC energy.
                    "smooth" => {
                        let v = (x * 4 + y * 3) >> 1;
                        (v as u8, (v / 2 + 20) as u8, (255 - v / 3) as u8)
                    }
                    // Texture: fine deterministic checker + high-frequency weave.
                    "texture" => {
                        let v = (((x ^ y) & 1) * 90) as u8
                            + (((x.wrapping_mul(y)) & 7) * 8) as u8;
                        (v, v.wrapping_add(40), v.wrapping_add(90))
                    }
                    // Text: glyph-like strokes on a flat background.
                    "text" => {
                        let on = (x % 9 < 2) || (y % 11 < 2) || ((x + y) % 17 < 2);
                        let v = if on { 20u8 } else { 235u8 };
                        (v, v, v)
                    }
                    // Noise: LCG white noise across all three channels.
                    "noise" => {
                        let s = (x as u32)
                            .wrapping_mul(374761393)
                            .wrapping_add((y as u32).wrapping_mul(668265263))
                            .wrapping_mul(1274126177);
                        let r = ((s >> 13) & 0xff) as u8;
                        let g = ((s.wrapping_mul(2246822519) >> 11) & 0xff) as u8;
                        let b = ((s.wrapping_mul(3266489917) >> 9) & 0xff) as u8;
                        (r, g, b)
                    }
                    _ => unreachable!(),
                };
                img[i] = r;
                img[i + 1] = g;
                img[i + 2] = b;
            }
        }
        img
    }

    // Automated mean-opinion-score *proxy* (not a human rating): a transparent
    // blend of PSNR and global SSIM, mapped to the 1..5 MOS scale.
    fn mos_proxy(p: f64, s: f64) -> f64 {
        let p_score = ((p - 20.0) / 25.0).clamp(0.0, 1.0); // 20 dB -> 0, 45 dB -> 1
        let s_score = ((s - 0.6) / 0.4).clamp(0.0, 1.0);
        (1.0 + 4.0 * (0.5 * p_score + 0.5 * s_score) * 100.0).round() / 100.0
    }

    #[test]
    fn four_material_evaluation_report() {
        const W: usize = 64;
        const H: usize = 64;
        const Q: u8 = 85;
        let weights = default_perceptual_weights();
        let custom: [f32; 64] = {
            let mut t = [1.0f32; 64];
            let mut z = 0;
            while z < 64 {
                t[z] = 1.0 + 2.0 * ((z as f32 / 63.0).powi(3));
                z += 1;
            }
            t
        };
        let custom_percent_len = CUSTOM_WEIGHT_TRAILER_LEN;

        println!();
        println!("================ perceptual weighted-quant evaluation (64x64 RGB, q={Q}) ================");
        println!(
            "{:<8} | {:>9} {:>9} {:>8} | {:>7} {:>7} {:>6} | {:>7} {:>7} {:>6} | {:>9} {:>9}",
            "material", "u_bytes", "w_bytes", "ratio", "u_psnr", "w_psnr", "d_psnr",
            "u_ssim", "w_ssim", "d_ssim", "u_mos*", "w_mos*"
        );
        for kind in ["smooth", "texture", "text", "noise"] {
            let img = make_material(kind, W, H);

            let t0 = Instant::now();
            let uniform = encode_uniform(&img, W, H, Q);
            let t_enc_u = t0.elapsed();
            let t1 = Instant::now();
            let weighted = encode_weighted(&img, W, H, Q, &weights);
            let t_enc_w = t1.elapsed();
            let t2 = Instant::now();
            let custom_blob = encode_weighted(&img, W, H, Q, &custom);
            let t_enc_c = t2.elapsed();
            let t3 = Instant::now();
            let dec_u = decode_to_rgb(&uniform, W, H);
            let t_dec_u = t3.elapsed();
            let t4 = Instant::now();
            let dec_w = decode_to_rgb(&weighted, W, H);
            let t_dec_w = t4.elapsed();
            let dec_c = decode_to_rgb(&custom_blob, W, H);

            let ratio = weighted.len() as f64 / uniform.len() as f64;
            let (pu, pw, pc) = (psnr(&img, &dec_u), psnr(&img, &dec_w), psnr(&img, &dec_c));
            let (su, sw, sc) = (ssim(&img, &dec_u), ssim(&img, &dec_w), ssim(&img, &dec_c));
            println!(
                "{:<8} | {:>9} {:>9} {:>8.3} | {:>7.2} {:>7.2} {:>6.2} | {:>7.4} {:>7.4} {:>6.4} | {:>9.2} {:>9.2}",
                kind,
                uniform.len(), weighted.len(), ratio,
                pu, pw, pc - pu,
                su, sw, sc - su,
                mos_proxy(pu, su), mos_proxy(pw, sw),
            );
            // Weighted mode must never inflate default-table streams.
            assert!(weighted.len() <= uniform.len(), "{kind}: weighted stream grew");
            // Smooth material: weighted mode must stay within 2 dB of uniform.
            if kind == "smooth" {
                assert!(pw >= pu - 2.0, "smooth weighted PSNR {pw} vs uniform {pu}");
            }
            let _ = (t_enc_u, t_enc_w, t_enc_c, t_dec_u, t_dec_w);
        }
        // Timing on a larger sample so the numbers are not pure noise.
        let big = sample_rgb(512, 512, 2024);
        let t = Instant::now();
        let bu = encode_uniform(&big, 512, 512, Q);
        let enc_u = t.elapsed();
        let t = Instant::now();
        let bw = encode_weighted(&big, 512, 512, Q, &weights);
        let enc_w = t.elapsed();
        let t = Instant::now();
        let _ = decode_to_rgb(&bu, 512, 512);
        let dec_u = t.elapsed();
        let t = Instant::now();
        let _ = decode_to_rgb(&bw, 512, 512);
        let dec_w = t.elapsed();
        println!(
            "timing 512x512: enc uniform={:?} weighted={:?} | dec uniform={:?} weighted={:?}",
            enc_u, enc_w, dec_u, dec_w
        );
        println!(
            "weight table overhead: default table = 0 stream bytes (rebuilt both ends); custom table = {custom_percent_len} bytes ({} header/flags + 64 u16 LE)",
            4
        );
        println!("(*) u_mos/w_mos are an automated PSNR+SSIM proxy on the 1-5 MOS scale, not human ratings.");
        println!("========================================================================================");
    }
}
