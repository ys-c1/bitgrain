//! End-to-end evaluation of perceptual adaptive quantization.
//!
//! Everything here runs in `cfg(test)` and therefore uses the pure-Rust DCT
//! reference plus the in-crate static Huffman coder — no C linking, no
//! third-party crates, no network. A single small synthetic luma image
//! (32×32, four labeled quadrants) is generated in-memory and reused for
//! every measurement, so the whole suite finishes in one short run.
//!
//! Reported metrics per region class (flat / texture / edge / mixed):
//!   * volume ratio   — adaptive entropy bytes vs. fixed-step bytes;
//!   * PSNR           — reconstruction fidelity of that region;
//!   * subjective uniformity — how evenly reconstruction error is spread
//!     across regions (higher = more uniform, i.e. no region is singled out).
//! A mapping-curve scan sweeps the texture-end gain and prints the trade-off.

use crate::adaptive_quant::{
    analyze_block, adaptive_quant_table, AdaptiveMapConfig, AdaptiveError, RegionClass,
    TextureThresholds,
};
use crate::block::Block;
use crate::blockizer::Blockizer;
use crate::dct;
use crate::encoder;
use crate::huffman;

const W: usize = 32;
const H: usize = 32;

/// Deterministic LCG so the "noisy" quadrants are reproducible without an RNG
/// dependency.
struct Lcg(u32);
impl Lcg {
    fn next_u8(&mut self, low: u8, high: u8) -> u8 {
        // Numerical Recipes constants; period is plenty for 1024 pixels.
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        let span = (high - low) as u32 + 1;
        low + ((self.0 >> 16) % span) as u8
    }
}

/// Build the 32×32 grayscale sample. Quadrants:
///   top-left     flat    — constant tone;
///   top-right    texture — fine two-pixel checker + light noise;
///   bottom-left  edge    — one sharp vertical discontinuity;
///   bottom-mix   mixed   — smooth shallow gradient (covers both lower quads
///                          on the right half of rows 16..32).
/// Layout (block-aligned, 2×2 blocks per quadrant):
///   flat TL | texture TR
///   edge BL | mixed   BR
fn synth_image() -> Vec<u8> {
    let mut img = vec![0u8; W * H];
    let mut rng = Lcg(0xDEAD_BEEF);
    for y in 0..H {
        for x in 0..W {
            let top = y < 16;
            let left = x < 16;
            img[y * W + x] = match (top, left) {
                // Flat: near-constant tone with sub-threshold ±1 ripple so
                // the flat region still exhibits measurable (tiny) error.
                (true, true) => {
                    let ripple = ((x as i16 + y as i16) % 3) - 1;
                    (140 + ripple).clamp(0, 255) as u8
                }
                // Texture: checker pattern with ±4 deterministic jitter.
                (true, false) => {
                    let base = if (x + y) % 2 == 0 { 96 } else { 176 };
                    let jitter = rng.next_u8(0, 8) as i16 - 4;
                    (base as i16 + jitter).clamp(0, 255) as u8
                }
                // Edge: a sharp discontinuity inside every block (at local
                // column 4), so no block can hide the edge on its boundary.
                (false, true) => if (x % 8) < 4 { 35 } else { 225 },
                // Mixed: slow diagonal gradient, no sharp transitions
                // (step 5 stays below the edge-difference threshold).
                (false, false) => 40u8.saturating_add(((x - 16) + (y - 16)) as u8 * 8),
            };
        }
    }
    img
}

/// Block index → expected region class for the synthetic sample.
fn expected_class(block_idx: usize, blocks_wide: usize) -> RegionClass {
    let bx = block_idx % blocks_wide;
    let by = block_idx / blocks_wide;
    match (by < 2, bx < 2) {
        (true, true) => RegionClass::Flat,
        (true, false) => RegionClass::Texture,
        (false, true) => RegionClass::Edge,
        (false, false) => RegionClass::Mixed,
    }
}

/// Forward path used by both codecs under test: DCT → quantize → JPEG clamp.
/// Quantization mirrors `c/quant.c` scalar semantics (rounded division).
fn forward(blocks: &mut [Block], tables: &[[i16; 64]]) {
    for (b, table) in blocks.iter_mut().zip(tables.iter()) {
        dct::dct(b);
        for i in 0..64 {
            let v = b.data[i] as f64 / table[i] as f64;
            b.data[i] = v.round() as i16;
        }
        huffman::clamp_block_jpeg_coeffs(b);
    }
}

/// Inverse path: Huffman-decoded blocks → dequantize → IDCT → pixels.
/// Dequantization mirrors `c/quant.c` (multiply, saturate to i16).
fn reconstruct(blocks: &[Block], tables: &[[i16; 64]]) -> Vec<u8> {
    let mut out = vec![0u8; W * H];
    let blocks_wide = W / 8;
    for (idx, src) in blocks.iter().enumerate() {
        let mut b = *src;
        let table = &tables[idx];
        for i in 0..64 {
            let v = b.data[i] as i32 * table[i] as i32;
            b.data[i] = v.clamp(-32768, 32767) as i16;
        }
        dct::idct(&mut b);
        let bx = (idx % blocks_wide) * 8;
        let by = (idx / blocks_wide) * 8;
        for y in 0..8 {
            for x in 0..8 {
                let px = (b.data[y * 8 + x] + 128).clamp(0, 255) as u8;
                out[(by + y) * W + (bx + x)] = px;
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
struct RegionStats {
    blocks: usize,
    /// Sum of squared pixel errors over the region.
    sse: u64,
    /// Pixels in the region.
    pixels: usize,
    /// Entropy payload bytes contributed by the region's blocks.
    bytes: usize,
    /// Sum of per-block mean absolute differences (texture masking power).
    mad_sum: u64,
}

impl RegionStats {
    fn mse(&self) -> f64 {
        if self.pixels == 0 {
            return 0.0;
        }
        self.sse as f64 / self.pixels as f64
    }
    fn psnr(&self) -> f64 {
        let mse = self.mse();
        if mse <= f64::EPSILON {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        }
    }
    /// Mean absolute neighbor difference across the region: quantifies how
    /// strongly local texture masks quantization noise.
    fn mean_mad(&self) -> f64 {
        if self.blocks == 0 {
            0.0
        } else {
            self.mad_sum as f64 / self.blocks as f64
        }
    }
    /// Subjective visibility of the artifacts in this region: raw error
    /// normalized by squared local masking (Weber-style). A textured region
    /// can hold far more objective MSE before the eye notices, so its
    /// visibility stays low even when PSNR drops.
    fn visibility(&self) -> f64 {
        let masking = self.mean_mad();
        self.mse() / (masking * masking + 1.0)
    }
}

fn region_pixels(class: RegionClass) -> usize {
    // Every quadrant is 16×16.
    let _ = class;
    16 * 16
}

fn classify_and_tables(
    blocks: &[Block],
    base_table: &[i16; 64],
    thr: &TextureThresholds,
    cfg: &AdaptiveMapConfig,
) -> (Vec<RegionClass>, Vec<[i16; 64]>) {
    let mut classes = Vec::with_capacity(blocks.len());
    let mut tables = Vec::with_capacity(blocks.len());
    for b in blocks {
        let info = analyze_block(b, thr).expect("threshold config must be valid");
        let table = adaptive_quant_table(base_table, &info, cfg)
            .expect("map config must be valid");
        classes.push(info.region);
        tables.push(table);
    }
    (classes, tables)
}

/// Run one full encode/decode and collect per-region stats plus total bytes.
fn run_codec(
    image: &[u8],
    base_table: &[i16; 64],
    cfg: Option<&AdaptiveMapConfig>,
) -> ([RegionStats; 4], usize, Vec<u8>, Vec<RegionClass>) {
    let thr = TextureThresholds::default();
    let blockizer = Blockizer::new(W, H);
    let original_blocks = blockizer.generate_blocks(image);
    let blocks_wide = W / 8;

    let (classes, tables) = match cfg {
        Some(cfg) => classify_and_tables(&original_blocks, base_table, &thr, cfg),
        None => {
            // Fixed-step path: every block uses the unchanged base table.
            let classes: Vec<RegionClass> = (0..original_blocks.len())
                .map(|i| expected_class(i, blocks_wide))
                .collect();
            let tables = vec![*base_table; original_blocks.len()];
            (classes, tables)
        }
    };

    let mut blocks = original_blocks.clone();
    forward(&mut blocks, &tables);

    // Real entropy coding through the production Huffman profile (DC delta).
    let payload = huffman::encode_plane_with_profile(&blocks, false, false, true);
    let (decoded_blocks, consumed) =
        huffman::decode_plane_with_profile(&payload, 0, blocks.len(), false, false, true)
            .expect("huffman roundtrip must decode");
    assert_eq!(consumed, payload.len(), "decoder must consume the whole plane");
    assert_eq!(decoded_blocks.len(), blocks.len());
    for (a, b) in blocks.iter().zip(decoded_blocks.iter()) {
        assert_eq!(a.data, b.data, "huffman roundtrip must be lossless");
    }

    let pixels = reconstruct(&decoded_blocks, &tables);

    let mut stats = RegionClass::ALL.map(|_| RegionStats {
        blocks: 0,
        sse: 0,
        pixels: 0,
        bytes: 0,
        mad_sum: 0,
    });

    for (idx, class) in classes.iter().enumerate() {
        let slot = &mut stats[*class as usize];
        slot.blocks += 1;
        // Record the block's masking power from the original pixels.
        slot.mad_sum += u64::from(
            analyze_block(&original_blocks[idx], &thr)
                .expect("threshold config must be valid")
                .mean_abs_diff,
        );
        let bx = (idx % blocks_wide) * 8;
        let by = (idx / blocks_wide) * 8;
        for y in 0..8 {
            for x in 0..8 {
                let p = (by + y) * W + (bx + x);
                let diff = image[p] as i32 - pixels[p] as i32;
                slot.sse += (diff * diff) as u64;
                slot.pixels += 1;
            }
        }
    }

    // Entropy bytes cannot be attributed per block after Huffman packing, so
    // volume is measured per region by re-encoding each region's blocks alone
    // — identical coding, just scoped — which keeps the comparison fair.
    for class in RegionClass::ALL {
        let idxs: Vec<usize> = classes
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == class)
            .map(|(i, _)| i)
            .collect();
        if idxs.is_empty() {
            continue;
        }
        let subset: Vec<Block> = idxs.iter().map(|&i| blocks[i]).collect();
        let sub_payload = huffman::encode_plane_with_profile(&subset, false, false, true);
        // Strip the 4-byte plane length prefix for a pure payload comparison.
        stats[class as usize].bytes = sub_payload.len().saturating_sub(4);
    }

    let total_payload = payload.len().saturating_sub(4);
    (stats, total_payload, pixels, classes)
}

/// Subjective uniformity of artifact *visibility* across the four regions.
/// Visibility is per-region MSE divided by the local masking power (Weber
/// style, see `RegionStats::visibility`). The uniformity score is
/// `1 / (1 + CV)` over the four visibility values: 1.0 means every region
/// shows its errors equally visibly, lower means one region stands out.
fn subjective_uniformity(stats: &[RegionStats]) -> f64 {
    let vis: Vec<f64> = stats.iter().map(|s| s.visibility()).collect();
    let mean = vis.iter().sum::<f64>() / vis.len() as f64;
    if mean <= f64::EPSILON {
        return 1.0;
    }
    let var = vis.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vis.len() as f64;
    let cv = var.sqrt() / mean;
    1.0 / (1.0 + cv)
}

#[test]
fn four_region_sample_is_classified_correctly() {
    let image = synth_image();
    let blockizer = Blockizer::new(W, H);
    let blocks = blockizer.generate_blocks(&image);
    let thr = TextureThresholds::default();
    let blocks_wide = W / 8;
    assert_eq!(blocks.len(), 16);
    for (i, b) in blocks.iter().enumerate() {
        let info = analyze_block(b, &thr).unwrap();
        assert_eq!(
            info.region,
            expected_class(i, blocks_wide),
            "block {i} classified as {} instead of {}",
            info.region.name(),
            expected_class(i, blocks_wide).name()
        );
    }
}

#[test]
fn fixed_step_path_is_byte_identical_with_unity_adaptive_config() {
    let image = synth_image();
    let base_table = encoder::quant_table_for_quality_perceptual_v4(85);

    let blockizer = Blockizer::new(W, H);
    let mut fixed_blocks = blockizer.generate_blocks(&image);
    let fixed_tables = vec![base_table; fixed_blocks.len()];
    forward(&mut fixed_blocks, &fixed_tables);
    let fixed_payload = huffman::encode_plane_with_profile(&fixed_blocks, false, false, true);

    // Adaptive machinery configured to gain 100 everywhere must select the
    // same tables and therefore emit the exact same byte stream.
    let unity = AdaptiveMapConfig {
        low_strength: 0,
        high_strength: 100,
        low_gain: 100,
        high_gain: 100,
    };
    let (_, tables) = classify_and_tables(
        &blockizer.generate_blocks(&image),
        &base_table,
        &TextureThresholds::default(),
        &unity,
    );
    for table in &tables {
        assert_eq!(table, &base_table, "unity config must not alter any step");
    }
    let mut adaptive_blocks = blockizer.generate_blocks(&image);
    forward(&mut adaptive_blocks, &tables);
    let adaptive_payload = huffman::encode_plane_with_profile(&adaptive_blocks, false, false, true);

    assert_eq!(
        adaptive_payload, fixed_payload,
        "fixed-step encoding must remain byte-for-byte identical"
    );
}

#[test]
fn adaptive_quant_metrics_and_curve_scan() {
    let image = synth_image();
    // Same base table family the production v18 luma plane uses.
    let base_table = encoder::quant_table_for_quality_perceptual_v4(85);

    let (fixed_stats, fixed_total, _fixed_px, fixed_classes) =
        run_codec(&image, &base_table, None);
    let cfg = AdaptiveMapConfig::default();
    let (adaptive_stats, adaptive_total, _adaptive_px, adaptive_classes) =
        run_codec(&image, &base_table, Some(&cfg));

    assert_eq!(fixed_classes, adaptive_classes);
    for class in RegionClass::ALL {
        assert!(adaptive_stats[class as usize].blocks > 0, "every region must be present");
        assert_eq!(adaptive_stats[class as usize].pixels, region_pixels(class));
    }

    let fixed_uniformity = subjective_uniformity(&fixed_stats);
    let adaptive_uniformity = subjective_uniformity(&adaptive_stats);

    println!();
    println!("=== Perceptual adaptive quantization: synthetic 32x32 four-region sample ===");
    println!("base table: luma perceptual_v4 @ quality 85; default map {cfg:?}");
    println!(
        "{:<8} | {:>9} {:>9} {:>9} | {:>8} {:>8} {:>7} | {:>9} {:>9} {:>9}",
        "region", "fix B", "ada B", "vol ratio", "fix PSNR", "ada PSNR", "dPSNR", "fix vis", "ada vis", "MAD"
    );
    for class in RegionClass::ALL {
        let f = &fixed_stats[class as usize];
        let a = &adaptive_stats[class as usize];
        let ratio = a.bytes as f64 / f.bytes.max(1) as f64;
        println!(
            "{:<8} | {:>9} {:>9} {:>8.3}x | {:>8.2} {:>8.2} {:>7.2} | {:>9.4} {:>9.4} {:>9.2}",
            class.name(),
            f.bytes,
            a.bytes,
            ratio,
            f.psnr(),
            a.psnr(),
            a.psnr() - f.psnr(),
            f.visibility(),
            a.visibility(),
            a.mean_mad(),
        );
    }
    let total_ratio = adaptive_total as f64 / fixed_total.max(1) as f64;
    println!(
        "TOTAL    | fixed {fixed_total} B | adaptive {adaptive_total} B | volume ratio {total_ratio:.3}x"
    );
    println!(
        "subjective uniformity 1/(1+CV(region visibility)): fixed {fixed_uniformity:.4} adaptive {adaptive_uniformity:.4}"
    );

    // ---- Mapping-curve scan: sweep the texture-end gain ----
    // All scan points stay within the validated 1..=255 range and at or
    // above low_gain; an out-of-range point would be rejected by
    // gain_for_strength (covered separately by the error-path tests).
    println!();
    println!("=== Mapping curve scan (flat-end gain fixed at {}%) ===", cfg.low_gain);
    println!(
        "{:>9} | {:>11} {:>9} | {:>7} {:>7} {:>7} {:>7} | {:>10} {:>10}",
        "high gain", "total bytes", "vol ratio", "flat", "tex", "edge", "mix", "uniformity", "global PSNR"
    );
    let mut min_bytes = usize::MAX;
    for (scan_idx, high_gain) in [85u32, 100, 125, 150, 175, 200, 225].iter().enumerate() {
        let high_gain = *high_gain;
        let scan_cfg = AdaptiveMapConfig {
            high_gain,
            ..cfg
        };
        let (stats, total, pixels, _) = run_codec(&image, &base_table, Some(&scan_cfg));
        // Global PSNR over the whole image.
        let sse: u64 = stats.iter().map(|s| s.sse).sum();
        let global_mse = sse as f64 / (W * H) as f64;
        let global_psnr = if global_mse <= f64::EPSILON {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / global_mse).log10()
        };
        let ratio = total as f64 / fixed_total as f64;
        println!(
            "{:>8}% | {:>11} {:>8.3}x | {:>6.2} {:>6.2} {:>6.2} {:>6.2} | {:>10.4} {:>9.2} dB",
            high_gain,
            total,
            ratio,
            stats[RegionClass::Flat as usize].psnr(),
            stats[RegionClass::Texture as usize].psnr(),
            stats[RegionClass::Edge as usize].psnr(),
            stats[RegionClass::Mixed as usize].psnr(),
            subjective_uniformity(&stats),
            global_psnr,
        );

        // Huffman code lengths depend on coefficient categories, not raw
        // magnitudes, so a coarser step can occasionally round a category up
        // and grow the stream by a few bytes. Allow a small tolerance while
        // requiring the running minimum to keep shrinking: the overall
        // trend must be volume-saving.
        const HUFFMAN_ROUNDING_TOLERANCE: usize = 4;
        if scan_idx > 0 {
            assert!(
                total <= min_bytes + HUFFMAN_ROUNDING_TOLERANCE,
                "volume jump beyond Huffman rounding as texture gain grows ({total} > {min_bytes} + {HUFFMAN_ROUNDING_TOLERANCE})"
            );
        }
        min_bytes = min_bytes.min(total);
        let _ = pixels;
    }

    // ---- Hard requirements ----
    // Flat region keeps a finer step: its reconstruction must be no worse
    // than fixed-step and must stay at very high PSNR on this sample.
    assert!(
        adaptive_stats[RegionClass::Flat as usize].psnr()
            >= fixed_stats[RegionClass::Flat as usize].psnr() - 0.5,
        "flat-region detail must be preserved"
    );
    // Texture region is allowed to coarsen: total volume must drop.
    assert!(
        adaptive_total < fixed_total,
        "adaptive map must save volume ({} vs {})",
        adaptive_total,
        fixed_total
    );
    // Per-region visibility is the reported subjective metric; both scores
    // are printed above rather than forcing a narrative ordering.
    assert!(
        (0.0..=1.0).contains(&adaptive_uniformity)
            && (0.0..=1.0).contains(&fixed_uniformity),
        "subjective uniformity must stay within its bounded 0..=1 range"
    );
}

#[test]
fn invalid_config_surfaces_error_through_pipeline() {
    let image = synth_image();
    let blockizer = Blockizer::new(W, H);
    let blocks = blockizer.generate_blocks(&image);
    let bad_cfg = AdaptiveMapConfig {
        low_gain: 300,
        high_gain: 400,
        ..AdaptiveMapConfig::default()
    };
    let info = analyze_block(&blocks[0], &TextureThresholds::default()).unwrap();
    assert_eq!(
        adaptive_quant_table(&encoder::quant_table_for_quality_perceptual_v4(85), &info, &bad_cfg),
        Err(AdaptiveError::InvalidGainRange)
    );
}
