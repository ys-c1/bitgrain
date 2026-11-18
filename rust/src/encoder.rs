use crate::bitstream;
use crate::block::Block;
use crate::blockizer::Blockizer;
use crate::colorspace;
use crate::dct;
use crate::entropy;
#[cfg(not(test))]
use crate::ffi::quantize_block;
use crate::huffman;
use crate::zigzag::ZIGZAG;
use rayon::prelude::*;
use std::time::Instant;
const BLOCK_TILE_SIZE: usize = 512;
const PARALLEL_BLOCKS_THRESHOLD: usize = 384;
const PARALLEL_PLANE_PIXELS_THRESHOLD: usize = 262_144;
const PARALLEL_IMAGE_PIXELS_THRESHOLD: usize = 262_144;

// ===========================================================================
// Rate-distortion optimization (RDO)
//
// Three stages:
//   1. Estimate rate (exact Huffman payload size) and distortion (plane SSE
//      after reference IDCT) on a ladder of candidate quantization steps.
//   2. Pick the step minimizing D + lambda * R with a bisection over lambda
//      driven by the configurable target bits-per-pixel.
//   3. The chosen step is the v4 perceptual quality index; it is written into
//      the standard 12-byte header quality byte, so any v18/v19 decoder can
//      reconstruct the exact quantizer tables without a format change.
//
// This module is pure Rust (reference DCT/IDCT, local scalar quantizer) so it
// also links under `cargo test`, where the C FFI symbols are unavailable.
// The final production encode still uses the normal C/SIMD path; only the
// quality value fed to it is replaced by the RDO decision.
// ===========================================================================

/// RDO configuration. Both knobs are user-configurable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdoConfig {
    /// Target bits per pixel of the final .bg stream (must be > 0).
    pub target_bpp: f64,
    /// Maximum lambda bisection rounds (clamped to 1..=32).
    pub max_iterations: u32,
}

/// One evaluated quantization step with its measured rate/distortion.
#[derive(Debug, Clone, Copy)]
pub struct RdoTrial {
    /// Quantization step as stored in the header quality byte (1..=100).
    pub quality: u8,
    /// Total encoded bytes (12-byte header + all Huffman plane payloads).
    pub bytes: u64,
    /// Achieved bits per pixel.
    pub bpp: f64,
    /// Sum of squared errors across all reconstructed planes.
    pub distortion: u64,
}

/// Outcome of an RDO run.
#[derive(Debug, Clone)]
pub struct RdoResult {
    /// Step selected by the Lagrangian criterion.
    pub selected: RdoTrial,
    /// Every evaluated step (sorted by ascending quality).
    pub trials: Vec<RdoTrial>,
    /// Lambda bisection rounds actually executed.
    pub iterations: u32,
    /// True when the achieved bpp is within +/-10% of the target.
    pub target_reached: bool,
    /// Relative gap |bpp-target|/target (0 when exactly reached).
    pub relative_gap: f64,
    /// Wall-clock time spent inside the RDO search.
    pub elapsed: std::time::Duration,
}

#[inline]
fn rdo_quantize(coef: i16, table_entry: i16) -> i16 {
    // Match the C SIMD quantizer (c/quant.c): float32 division rounded to the
    // nearest integer (default MXCSR / vcvt rounding, ties to even). The
    // scalar C fallback truncates, but SSE2/AVX2/NEON production builds all
    // round, so the rate estimate must follow the rounding rule.
    ((coef as f32) / (table_entry as f32)).round_ties_even() as i16
}

#[inline]
fn rdo_dequantize(coef: i16, table_entry: i16) -> i16 {
    let v = coef as i32 * table_entry as i32;
    v.clamp(-32768, 32767) as i16
}

/// Encode one plane for a candidate step and return (payload bytes, SSE vs the
/// original plane). The input blocks are never mutated.
fn rdo_eval_plane(
    src_blocks: &[Block],
    table: &[i16; 64],
    sparsify: &[i16; 64],
    plane_w: usize,
    plane_h: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
    original: &[u8],
) -> (u64, u64) {
    let bw = (plane_w + 7) / 8;
    let mut work: Vec<Block> = Vec::with_capacity(src_blocks.len());
    let mut sse: u64 = 0;
    for src in src_blocks {
        let mut dct_block = *src;
        dct::dct(&mut dct_block); // reference DCT under cfg(test), C/SIMD in release
        let mut q = Block::new();
        for i in 0..64 {
            q.data[i] = rdo_quantize(dct_block.data[i], table[i]);
        }
        sparsify_ac_block(&mut q, sparsify);
        huffman::clamp_block_jpeg_coeffs(&mut q);

        // Reconstruct for distortion: dequant -> reference IDCT -> clamp to
        // the unsigned plane domain and compare against the source pixels.
        let mut r = Block::new();
        for i in 0..64 {
            r.data[i] = rdo_dequantize(q.data[i], table[i]);
        }
        r.data = dct::idct_reference(&r.data);
        for i in 0..64 {
            let recon = (r.data[i] + 128).clamp(0, 255) as u8;
            let by = (work.len() / bw) * 8 + i / 8;
            let bx = (work.len() % bw) * 8 + i % 8;
            if by < plane_h && bx < plane_w {
                let d = recon as i32 - original[by * plane_w + bx] as i32;
                sse += (d * d) as u64;
            }
        }
        work.push(q);
    }
    let payload = huffman::encode_plane_with_profile(
        &work, is_chroma, use_chroma_ac, use_dc_delta,
    );
    (payload.len() as u64, sse)
}

/// Run one candidate step over the Y/Cb/Cr (+ optional alpha) planes.
fn rdo_run_trial(
    quality: u8,
    planes: &RdoPlanes,
) -> RdoTrial {
    let luma_table = quant_table_for_quality_perceptual_v4(quality);
    let chroma_table = chroma_quant_table_for_quality_perceptual_v4(quality);
    let luma_sp = build_sparsify_thresholds(quality, false);
    let chroma_sp = build_sparsify_thresholds(quality, true);

    let (y_bytes, y_sse) = rdo_eval_plane(
        &planes.y_blocks, &luma_table, &luma_sp,
        planes.width, planes.height, false, false, true, &planes.y,
    );
    let cw = (planes.width + 1) / 2;
    let ch = (planes.height + 1) / 2;
    let (cb_bytes, cb_sse) = rdo_eval_plane(
        &planes.cb_blocks, &chroma_table, &chroma_sp,
        cw, ch, true, true, true, &planes.cb,
    );
    let (cr_bytes, cr_sse) = rdo_eval_plane(
        &planes.cr_blocks, &chroma_table, &chroma_sp,
        cw, ch, true, true, true, &planes.cr,
    );
    let (a_bytes, a_sse) = match (&planes.a_blocks, &planes.a) {
        (Some(ab), Some(ap)) => rdo_eval_plane(
            ab, &luma_table, &luma_sp,
            planes.width, planes.height, false, false, true, ap,
        ),
        _ => (0, 0),
    };

    let bytes = BG_HEADER_SIZE as u64 + y_bytes + cb_bytes + cr_bytes + a_bytes;
    let bpp = (bytes * 8) as f64 / planes.pixels as f64;
    RdoTrial {
        quality,
        bytes,
        bpp,
        distortion: y_sse + cb_sse + cr_sse + a_sse,
    }
}

struct RdoPlanes {
    width: usize,
    height: usize,
    pixels: usize,
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    a: Option<Vec<u8>>,
    y_blocks: Vec<Block>,
    cb_blocks: Vec<Block>,
    cr_blocks: Vec<Block>,
    a_blocks: Option<Vec<Block>>,
}

fn rdo_build_rgb_planes(image: &[u8], width: usize, height: usize) -> RdoPlanes {
    let (y, cb, cr) = colorspace::rgb_to_ycbcr420(image, width, height);
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let bf = Blockizer::new(width, height);
    let bc = Blockizer::new(cw, ch);
    RdoPlanes {
        width,
        height,
        pixels: width * height,
        y_blocks: bf.generate_blocks(&y),
        cb_blocks: bc.generate_blocks(&cb),
        cr_blocks: bc.generate_blocks(&cr),
        a_blocks: None,
        y,
        cb,
        cr,
        a: None,
    }
}

fn rdo_build_rgba_planes(image: &[u8], width: usize, height: usize) -> RdoPlanes {
    let (y, cb, cr, a) = colorspace::rgba_to_ycbcr420a(image, width, height);
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;
    let bf = Blockizer::new(width, height);
    let bc = Blockizer::new(cw, ch);
    RdoPlanes {
        width,
        height,
        pixels: width * height,
        y_blocks: bf.generate_blocks(&y),
        cb_blocks: bc.generate_blocks(&cb),
        cr_blocks: bc.generate_blocks(&cr),
        a_blocks: Some(bf.generate_blocks(&a)),
        y,
        cb,
        cr,
        a: Some(a),
    }
}

/// Uniform step ladder covering the full legal quality range. The endpoints
/// (1 = coarsest, 100 = finest) bound every achievable rate.
fn rdo_candidate_ladder() -> Vec<u8> {
    let mut v: Vec<u8> = (1..=100u8).collect();
    v.dedup();
    v
}

/// Stage 1 + 2: evaluate the ladder and bisect lambda against the target bpp.
fn rdo_search(planes: &RdoPlanes, config: RdoConfig) -> RdoResult {
    let start = Instant::now();
    let max_iters = config.max_iterations.clamp(1, 32);

    // Stage 1: rate/distortion estimation for every quantization step.
    let mut trials: Vec<RdoTrial> = rdo_candidate_ladder()
        .iter()
        .map(|&q| rdo_run_trial(q, planes))
        .collect();
    trials.sort_by_key(|t| t.quality);

    let target_bytes = config.target_bpp * planes.pixels as f64 / 8.0;
    let pick = |lambda: f64| -> usize {
        let mut best = 0usize;
        let mut best_cost = f64::INFINITY;
        for (i, t) in trials.iter().enumerate() {
            let cost = t.distortion as f64 + lambda * t.bytes as f64;
            if cost < best_cost {
                best_cost = cost;
                best = i;
            }
        }
        best
    };

    // Stage 2: bisection on lambda. Higher lambda favors fewer bytes, so the
    // selected byte count decreases monotonically as lambda grows.
    let mut lo = 0.0f64;
    let mut hi = 1.0e12f64;
    let mut iterations = 0u32;
    let mut chosen = pick(lo);
    let mut prev_chosen = usize::MAX;
    let mut stable = 0u32;
    while iterations < max_iters {
        iterations += 1;
        let mid = if lo == 0.0 && hi == 0.0 { 0.0 } else { (lo + hi) / 2.0 };
        chosen = pick(mid);
        if chosen == prev_chosen {
            stable += 1;
            if stable >= 2 {
                break;
            }
        } else {
            stable = 0;
        }
        prev_chosen = chosen;
        if trials[chosen].bytes as f64 > target_bytes {
            lo = mid; // too large: penalize rate harder
        } else {
            hi = mid;
        }
        if (hi - lo).abs() <= 1.0e-3 * hi.max(1.0) {
            break;
        }
    }

    // Nearest achievable step to the target, used only when no trial lands
    // within the +/-10% reachability band. Ties prefer the finer step.
    let nearest = trials
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (a.bytes as f64 - target_bytes)
                .abs()
                .partial_cmp(&(b.bytes as f64 - target_bytes).abs())
                .unwrap()
                .then_with(|| b.quality.cmp(&a.quality))
        })
        .map(|(i, _)| i)
        .unwrap_or(chosen);
    let within_band = |i: usize| -> bool {
        (trials[i].bpp - config.target_bpp).abs() <= 0.10 * config.target_bpp
    };

    // Decision:
    //  * target reachable  -> keep the Lagrangian (bisection) choice; if the
    //    lambda search stopped just outside the band, take the nearest in-band
    //    step so a reachable target is always reported as reached.
    //  * target unreachable -> nearest achievable step, gap reported to caller.
    let chosen = if within_band(nearest) {
        if within_band(chosen) { chosen } else { nearest }
    } else {
        nearest
    };

    let gap = (trials[chosen].bpp - config.target_bpp).abs() / config.target_bpp;
    RdoResult {
        selected: trials[chosen],
        trials,
        iterations,
        target_reached: gap <= 0.10,
        relative_gap: gap,
        elapsed: start.elapsed(),
    }
}

/// RDO for an interleaved RGB image (v18-compatible output step).
pub fn rdo_encode_rgb(
    image: &[u8],
    width: usize,
    height: usize,
    config: RdoConfig,
) -> RdoResult {
    let planes = rdo_build_rgb_planes(image, width, height);
    rdo_search(&planes, config)
}

/// RDO for an interleaved RGBA image (v19-compatible output step).
pub fn rdo_encode_rgba(
    image: &[u8],
    width: usize,
    height: usize,
    config: RdoConfig,
) -> RdoResult {
    let planes = rdo_build_rgba_planes(image, width, height);
    rdo_search(&planes, config)
}

/// Parse RDO knobs from the environment. Returns None when RDO is disabled or
/// the environment is malformed, leaving the fixed-step path untouched.
///   BITGRAIN_RDO_BPP   - target bits per pixel (> 0)
///   BITGRAIN_RDO_ITERS - max lambda bisection rounds (default 6, 1..=32)
pub fn rdo_config_from_env() -> Option<RdoConfig> {
    let bpp = std::env::var("BITGRAIN_RDO_BPP").ok()?;
    let target_bpp = bpp.trim().parse::<f64>().ok()?;
    if !target_bpp.is_finite() || target_bpp <= 0.0 {
        return None;
    }
    let max_iterations = std::env::var("BITGRAIN_RDO_ITERS")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(6)
        .clamp(1, 32);
    Some(RdoConfig { target_bpp, max_iterations })
}

/// Apply an RDO decision to a fixed-step quality input: when RDO is enabled
/// via the environment, replace the step with the Lagrangian choice and log
/// the outcome on stderr. Otherwise the original quality passes through
/// byte-identically.
fn rdo_quality_for_rgb(width: usize, height: usize, image: &[u8], quality: u8) -> u8 {
    let Some(cfg) = rdo_config_from_env() else { return quality };
    match std::panic::catch_unwind(|| rdo_encode_rgb(image, width, height, cfg)) {
        Ok(res) => {
            eprintln!(
                "RDO: target {:.3} bpp -> step {} ({:.3} bpp, gap {:+.1}%, {} iters, {:.2} ms){}",
                cfg.target_bpp,
                res.selected.quality,
                res.selected.bpp,
                res.relative_gap * 100.0,
                res.iterations,
                res.elapsed.as_secs_f64() * 1000.0,
                if res.target_reached { "" } else { " [target unreachable: nearest step used]" },
            );
            res.selected.quality
        }
        Err(_) => quality,
    }
}

fn rdo_quality_for_rgba(width: usize, height: usize, image: &[u8], quality: u8) -> u8 {
    let Some(cfg) = rdo_config_from_env() else { return quality };
    match std::panic::catch_unwind(|| rdo_encode_rgba(image, width, height, cfg)) {
        Ok(res) => {
            eprintln!(
                "RDO: target {:.3} bpp -> step {} ({:.3} bpp, gap {:+.1}%, {} iters, {:.2} ms){}",
                cfg.target_bpp,
                res.selected.quality,
                res.selected.bpp,
                res.relative_gap * 100.0,
                res.iterations,
                res.elapsed.as_secs_f64() * 1000.0,
                if res.target_reached { "" } else { " [target unreachable: nearest step used]" },
            );
            res.selected.quality
        }
        Err(_) => quality,
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

#[inline]
pub fn quantize(block: &mut [i16; 64], table: &[i16; 64]) {
    // Release builds use the C/SIMD quantizer; test builds use the equivalent
    // rounding rule (see rdo_quantize / c/quant.c) so the full pipeline links
    // under `cargo test` and mirrors production rate behavior.
    #[cfg(not(test))]
    unsafe { quantize_block(block.as_mut_ptr(), table.as_ptr()); }
    #[cfg(test)]
    for i in 0..64 {
        block[i] = rdo_quantize(block[i], table[i]);
    }
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
    // RDO (opt-in via BITGRAIN_RDO_BPP): replace the fixed step with the
    // Lagrangian choice. No env var -> identical fixed-step behavior.
    let quality = rdo_quality_for_rgb(width, height, image, quality);
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
    // RDO (opt-in via BITGRAIN_RDO_BPP): replace the fixed step with the
    // Lagrangian choice. No env var -> identical fixed-step behavior.
    let quality = rdo_quality_for_rgba(width, height, image, quality);
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
// RDO unit tests (pure-Rust path; no C FFI linkage required under cargo test)
// ===========================================================================

#[cfg(test)]
mod rdo_tests {
    use super::*;

    const W: usize = 48;
    const H: usize = 48;
    const NPIX: usize = W * H;

    fn make_flat() -> Vec<u8> {
        vec![90u8, 140, 200].repeat(NPIX)
    }

    fn make_gradient() -> Vec<u8> {
        let mut img = Vec::with_capacity(NPIX * 3);
        for y in 0..H {
            for x in 0..W {
                img.push((x * 255 / W) as u8);
                img.push((y * 255 / H) as u8);
                img.push(((x + y) / 2).min(255) as u8);
            }
        }
        img
    }

    /// Deterministic LCG pseudo-noise (no external RNG dependency).
    fn make_noise() -> Vec<u8> {
        let mut state = 0x1234_5678u32;
        let mut img = Vec::with_capacity(NPIX * 3);
        for _ in 0..NPIX * 3 {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            img.push((state >> 16) as u8);
        }
        img
    }

    /// Hard 8x8 checker blocks: rich high-frequency content.
    fn make_blocks() -> Vec<u8> {
        let mut img = Vec::with_capacity(NPIX * 3);
        for y in 0..H {
            for x in 0..W {
                let v = if ((x / 8) + (y / 8)) % 2 == 0 { 230u8 } else { 24u8 };
                img.extend_from_slice(&[v, v, 255 - v]);
            }
        }
        img
    }

    fn psnr_from_sse(sse: u64, samples: usize) -> f64 {
        let mse = sse as f64 / samples as f64;
        if mse <= 0.0 { 99.0 } else { 10.0 * (255.0 * 255.0 / mse).log10() }
    }

    #[test]
    fn rdo_covers_all_100_quantization_steps() {
        let img = make_gradient();
        let res = rdo_encode_rgb(&img, W, H, RdoConfig { target_bpp: 2.0, max_iterations: 6 });
        assert_eq!(res.trials.len(), 100, "every quantization step must be evaluated");
        for (a, b) in res.trials.iter().zip(res.trials.iter().skip(1)) {
            assert!(a.quality < b.quality);
            // Rounding quantizers can invert adjacent steps by a few bytes;
            // bound any inversion instead of demanding strict monotonicity.
            assert!(
                a.bytes <= b.bytes + 4,
                "adjacent steps must not invert by more than a few bytes ({} -> {})",
                a.bytes, b.bytes
            );
        }
        assert!(res.iterations >= 1 && res.iterations <= 6);
        assert!(res.selected.quality >= 1 && res.selected.quality <= 100);
    }

    #[test]
    fn rdo_reaches_reasonable_target_and_embeds_step() {
        let img = make_gradient();
        // 0.8 bpp sits inside this image's achievable range (~0.23..1.53 bpp).
        let cfg = RdoConfig { target_bpp: 0.8, max_iterations: 6 };
        let res = rdo_encode_rgb(&img, W, H, cfg);
        assert!(res.target_reached, "reachable target should be within +/-10%, got gap {:.2}%", res.relative_gap * 100.0);

        // The selected step must be embedded in the header quality byte, and
        // the produced stream must be byte-identical to the fixed-step encode
        // at that exact quality (same v18 pipeline, no format change).
        let mut rdo_out = vec![0u8; NPIX * 3 * 2 + 4096];
        let mut rdo_pos = 0i32;
        encode_rgb_ycbcr(&img, W, H, res.selected.quality, &mut rdo_out, &mut rdo_pos, None);
        assert_eq!(rdo_out[11], res.selected.quality);
        assert_eq!(rdo_pos as u64, res.selected.bytes, "estimated size must match the real stream size");

        let mut ref_out = vec![0u8; NPIX * 3 * 2 + 4096];
        let mut ref_pos = 0i32;
        encode_rgb_ycbcr(&img, W, H, res.selected.quality, &mut ref_out, &mut ref_pos, None);
        assert_eq!(&rdo_out[..rdo_pos as usize], &ref_out[..ref_pos as usize]);
    }

    #[test]
    fn rdo_unreachable_target_returns_nearest_step_with_gap() {
        let img = make_flat();
        // A target above the finest-step achievable rate cannot be matched.
        let cfg = RdoConfig { target_bpp: 24.0, max_iterations: 4 };
        let res = rdo_encode_rgb(&img, W, H, cfg);
        assert!(!res.target_reached);
        assert!(res.relative_gap > 0.10);
        // Nearest achievable result is the finest step.
        assert_eq!(res.selected.quality, 100);
    }

    #[test]
    fn rdo_is_deterministic() {
        let img = make_noise();
        let cfg = RdoConfig { target_bpp: 1.5, max_iterations: 5 };
        let a = rdo_encode_rgb(&img, W, H, cfg);
        let b = rdo_encode_rgb(&img, W, H, cfg);
        assert_eq!(a.selected.quality, b.selected.quality);
        assert_eq!(a.selected.bytes, b.selected.bytes);
        assert_eq!(a.selected.distortion, b.selected.distortion);
    }

    #[test]
    fn rdo_rgba_path_supports_alpha() {
        let mut img = make_blocks();
        // Inject an alpha channel: gradient transparency.
        let mut rgba = Vec::with_capacity(NPIX * 4);
        for (i, px) in img.chunks_exact_mut(3).enumerate() {
            rgba.extend_from_slice(px);
            rgba.push((i % 256) as u8);
        }
        let cfg = RdoConfig { target_bpp: 3.0, max_iterations: 5 };
        let res = rdo_encode_rgba(&rgba, W, H, cfg);
        assert_eq!(res.trials.len(), 100);
        assert!(res.selected.quality >= 1);
        let mut out = vec![0u8; NPIX * 4 * 2 + 4096];
        let mut pos = 0i32;
        encode_rgba_ycbcr(&rgba, W, H, res.selected.quality, &mut out, &mut pos, None);
        assert_eq!(out[2], 0x13, "RGBA path must keep the v19 magic");
        assert_eq!(out[11], res.selected.quality);
        assert_eq!(pos as u64, res.selected.bytes);
    }

    #[test]
    fn rdo_four_materials_produce_metrics() {
        // Exact-material sanity + PSNR reporting used by the evaluation script.
        let materials = [
            ("flat", make_flat()),
            ("gradient", make_gradient()),
            ("noise", make_noise()),
            ("blocks", make_blocks()),
        ];
        let cfg = RdoConfig { target_bpp: 2.0, max_iterations: 6 };
        // Samples counted over the three Y/Cb/Cr planes.
        let samples = NPIX + 2 * ((W + 1) / 2) * ((H + 1) / 2);
        for (name, img) in materials.iter() {
            let res = rdo_encode_rgb(img, W, H, cfg);
            let psnr = psnr_from_sse(res.selected.distortion, samples);
            assert!(psnr.is_finite() && psnr > 0.0, "{name}: bad PSNR {psnr}");
            assert!(res.selected.bpp > 0.0);
            assert!(res.elapsed.as_nanos() > 0);
            assert!(res.iterations >= 1);
        }
    }

    #[test]
    fn rdo_config_validation() {
        assert!(RdoConfig { target_bpp: 0.0, max_iterations: 6 }.target_bpp == 0.0);
        // Ladder endpoints must stay in the legal header range.
        assert_eq!(*rdo_candidate_ladder().first().unwrap(), 1u8);
        assert_eq!(*rdo_candidate_ladder().last().unwrap(), 100u8);
    }
}
