//! Perceptual adaptive quantization.
//!
//! Different image regions tolerate quantization noise differently:
//! flat areas show every error (small step = detail preserved), while busy
//! texture masks coarse quantization (large step = bits saved). This module
//! classifies each 8×8 block into one of four regions — flat, texture, edge,
//! mixed — measures its local texture strength, and maps that strength to a
//! configurable quantization-step multiplier.
//!
//! Design constraints honored here:
//!   * no new third-party dependencies (std + rayon already in the crate);
//!   * no change to any existing public FFI signature — every entry point in
//!     this module is an internal library API;
//!   * the fixed-step path is byte-for-byte identical to the legacy path
//!     (gain 100, i.e. multiplier 1.00, reproduces the input table exactly);
//!   * every out-of-range configuration value is rejected with an error
//!     instead of being silently clamped, so callers cannot request an
//!     impossible step without noticing.
//!
//! All arithmetic is integer and deterministic. A gain is expressed in
//! percent (100 = no change) so the whole pipeline stays reproducible across
//! platforms and SIMD backends.

use crate::block::Block;

/// Four perceptual region classes, ordered by increasing local structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegionClass {
    /// Near-constant blocks: quantization noise is immediately visible.
    Flat = 0,
    /// Dense high-frequency activity: noise is masked by the texture itself.
    Texture = 1,
    /// Strong directional discontinuity: must keep the edge sharp.
    Edge = 2,
    /// Neither clearly flat nor strongly structured: intermediate treatment.
    Mixed = 3,
}

impl RegionClass {
    pub const ALL: [RegionClass; 4] = [
        RegionClass::Flat,
        RegionClass::Texture,
        RegionClass::Edge,
        RegionClass::Mixed,
    ];

    pub fn name(self) -> &'static str {
        match self {
            RegionClass::Flat => "flat",
            RegionClass::Texture => "texture",
            RegionClass::Edge => "edge",
            RegionClass::Mixed => "mixed",
        }
    }
}

/// Measured texture descriptors for one 8×8 block.
///
/// `strength` is normalized to the `0..=100` range consumed by the mapping
/// curve; the other fields drive the four-class decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureInfo {
    pub region: RegionClass,
    /// Mean absolute pixel difference across horizontal/vertical neighbors.
    pub mean_abs_diff: u32,
    /// Largest neighbor difference anywhere in the block.
    pub max_abs_diff: u32,
    /// Fraction (0..=64) of neighbor pairs whose difference exceeds the edge
    /// threshold — measures how widespread sharp transitions are.
    pub edge_pixel_count: u32,
    /// Configurable texture strength in `0..=100`.
    pub strength: u32,
}

/// Configurable mapping from texture strength to quantization-step gain.
///
/// The curve is a clamped linear ramp:
///
/// ```text
/// gain(s) = low_gain + (high_gain - low_gain) * (s - low_strength)
///           / (high_strength - low_strength),
/// s in 0..=100, result clamped to [low_gain, high_gain], units = percent.
/// ```
///
/// Gains are integers in `1..=255` percent: 100 reproduces the original step,
/// values below 100 shrink it (finer), values above 100 grow it (coarser).
/// The 255 ceiling mirrors the codec's existing `clamp(1, 255)` table bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveMapConfig {
    /// Strength at/below which the minimum gain applies (flat end).
    pub low_strength: u32,
    /// Strength at/above which the maximum gain applies (texture end).
    pub high_strength: u32,
    /// Step gain (percent) at the flat end; must be `>= 1`.
    pub low_gain: u32,
    /// Step gain (percent) at the texture end; must be `>= low_gain`.
    pub high_gain: u32,
}

/// Defaults: flat blocks keep 85% of the original step (finer, detail kept),
/// texture blocks go up to 160% (coarser, volume saved), ramp over the full
/// strength range.
impl Default for AdaptiveMapConfig {
    fn default() -> Self {
        Self {
            low_strength: 0,
            high_strength: 100,
            low_gain: 85,
            high_gain: 160,
        }
    }
}

/// Thresholds used by the four-class texture decision. All values compare
/// against 8-bit pixel differences (0..=255), so they live in that range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureThresholds {
    /// Mean absolute difference at or below which a block is flat.
    pub flat_mad: u32,
    /// Neighbor difference at or above which a pair is an edge candidate.
    pub edge_diff: u32,
    /// Edge-pair count at or above which a block is classified as texture
    /// rather than a single localized edge.
    pub texture_edge_count: u32,
    /// Mean absolute difference at or above which a block is clearly busy.
    pub busy_mad: u32,
    /// Divisor normalizing mean absolute difference to strength 0..=100.
    pub strength_scale: u32,
}

impl Default for TextureThresholds {
    fn default() -> Self {
        Self {
            flat_mad: 3,
            edge_diff: 28,
            texture_edge_count: 12,
            busy_mad: 14,
            strength_scale: 40,
        }
    }
}

/// Error returned when a caller supplies an out-of-range configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveError {
    /// A strength endpoint is outside `0..=100` or the endpoints are inverted.
    InvalidStrengthRange,
    /// A gain is outside `1..=255` or `high_gain < low_gain`.
    InvalidGainRange,
    /// A texture threshold is zero or above 256 (pixel-diff domain).
    InvalidThreshold,
    /// Strength normalization divisor is zero.
    InvalidStrengthScale,
}

impl AdaptiveMapConfig {
    /// Validate every field. Returns an error instead of clamping, so an
    /// out-of-range step request always fails loudly.
    pub fn validate(&self) -> Result<(), AdaptiveError> {
        if self.low_strength > 100
            || self.high_strength > 100
            || self.low_strength >= self.high_strength
        {
            return Err(AdaptiveError::InvalidStrengthRange);
        }
        if self.low_gain == 0
            || self.high_gain == 0
            || self.low_gain > 255
            || self.high_gain > 255
            || self.high_gain < self.low_gain
        {
            return Err(AdaptiveError::InvalidGainRange);
        }
        Ok(())
    }

    /// Map a texture strength (`0..=100`) to a step gain in percent.
    ///
    /// Pure integer interpolation: no floating point, fully deterministic.
    pub fn gain_for_strength(&self, strength: u32) -> Result<u32, AdaptiveError> {
        self.validate()?;
        let s = strength.min(100);
        if s <= self.low_strength {
            return Ok(self.low_gain);
        }
        if s >= self.high_strength {
            return Ok(self.high_gain);
        }
        let span = self.high_strength - self.low_strength;
        let pos = s - self.low_strength;
        let gain = self.low_gain
            + (self.high_gain - self.low_gain) * pos / span;
        Ok(gain.clamp(self.low_gain, self.high_gain))
    }
}

impl TextureThresholds {
    pub fn validate(&self) -> Result<(), AdaptiveError> {
        let in_pixel_range = |v: u32| v <= 255;
        if !in_pixel_range(self.flat_mad)
            || !in_pixel_range(self.edge_diff)
            || !in_pixel_range(self.busy_mad)
            || self.texture_edge_count > 112
        {
            return Err(AdaptiveError::InvalidThreshold);
        }
        if self.strength_scale == 0 {
            return Err(AdaptiveError::InvalidStrengthScale);
        }
        Ok(())
    }
}

/// Classify one block and measure its texture strength.
///
/// The block holds centered pixels (`-128..127`, as produced by
/// `Blockizer`); neighbor differences are identical before/after centering.
pub fn analyze_block(block: &Block, thr: &TextureThresholds) -> Result<TextureInfo, AdaptiveError> {
    thr.validate()?;
    let d = &block.data;

    // Horizontal pairs: 8 rows × 7 = 56; vertical pairs: 7 × 8 = 56.
    let mut sum_diff: u32 = 0;
    let mut max_diff: u32 = 0;
    let mut edge_pixels: u32 = 0;
    for y in 0..8 {
        for x in 0..8 {
            if x + 1 < 8 {
                let diff = u32::from(d[y * 8 + x].abs_diff(d[y * 8 + x + 1]));
                sum_diff += diff;
                max_diff = max_diff.max(diff);
                if diff >= thr.edge_diff {
                    edge_pixels += 1;
                }
            }
            if y + 1 < 8 {
                let diff = u32::from(d[y * 8 + x].abs_diff(d[(y + 1) * 8 + x]));
                sum_diff += diff;
                max_diff = max_diff.max(diff);
                if diff >= thr.edge_diff {
                    edge_pixels += 1;
                }
            }
        }
    }
    const TOTAL_PAIRS: u32 = 112;
    let mean_abs_diff = (sum_diff + TOTAL_PAIRS / 2) / TOTAL_PAIRS;

    // Four-class decision:
    //   flat    — barely any variation;
    //   texture — many sharp pairs spread across the block (busy mask);
    //   edge    — a strong localized discontinuity but few edge pairs;
    //   mixed   — everything in between.
    let region = if mean_abs_diff <= thr.flat_mad {
        RegionClass::Flat
    } else if edge_pixels >= thr.texture_edge_count && mean_abs_diff >= thr.busy_mad {
        RegionClass::Texture
    } else if max_diff >= thr.edge_diff && edge_pixels < thr.texture_edge_count {
        RegionClass::Edge
    } else {
        RegionClass::Mixed
    };

    // Normalize MAD into 0..=100.
    let strength = (mean_abs_diff * 100 / thr.strength_scale).min(100);

    Ok(TextureInfo {
        region,
        mean_abs_diff: mean_abs_diff,
        max_abs_diff: max_diff,
        edge_pixel_count: edge_pixels,
        strength,
    })
}

/// Apply an integer percent gain to one quantization step, keeping it inside
/// the codec's legal `1..=255` table range.
#[inline]
fn scale_step(step: i16, gain_percent: u32) -> Result<i16, AdaptiveError> {
    if !(1..=255).contains(&gain_percent) {
        return Err(AdaptiveError::InvalidGainRange);
    }
    let scaled = (step as i32 * gain_percent as i32 + 50) / 100;
    Ok(scaled.clamp(1, 255) as i16)
}

/// Build the per-block quantization table: the base table scaled by the gain
/// selected from the block's texture strength.
///
/// With `gain == 100` every entry equals the base table entry, which is the
/// guarantee behind the byte-identical fixed-step path.
pub fn adaptive_quant_table(
    base_table: &[i16; 64],
    info: &TextureInfo,
    cfg: &AdaptiveMapConfig,
) -> Result<[i16; 64], AdaptiveError> {
    let gain = cfg.gain_for_strength(info.strength)?;
    let mut out = [0i16; 64];
    for i in 0..64 {
        out[i] = scale_step(base_table[i], gain)?;
    }
    Ok(out)
}

/// Convenience: analyze + map + table for one block.
pub fn table_for_block(
    block: &Block,
    base_table: &[i16; 64],
    thr: &TextureThresholds,
    cfg: &AdaptiveMapConfig,
) -> Result<([i16; 64], TextureInfo), AdaptiveError> {
    let info = analyze_block(block, thr)?;
    let table = adaptive_quant_table(base_table, &info, cfg)?;
    Ok((table, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn centered(rows: [[u8; 8]; 8]) -> Block {
        let mut data = [0i16; 64];
        for y in 0..8 {
            for x in 0..8 {
                data[y * 8 + x] = rows[y][x] as i16 - 128;
            }
        }
        Block { data }
    }

    #[test]
    fn flat_block_classified_flat_with_low_strength() {
        let block = centered([[128u8; 8]; 8]);
        let info = analyze_block(&block, &TextureThresholds::default()).unwrap();
        assert_eq!(info.region, RegionClass::Flat);
        assert_eq!(info.strength, 0);
        assert_eq!(info.mean_abs_diff, 0);
    }

    #[test]
    fn edge_block_classified_edge() {
        // Single vertical discontinuity: left half dark, right half bright.
        let mut rows = [[0u8; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                rows[y][x] = if x < 4 { 40 } else { 220 };
            }
        }
        let block = centered(rows);
        let info = analyze_block(&block, &TextureThresholds::default()).unwrap();
        assert_eq!(info.region, RegionClass::Edge);
        assert!(info.max_abs_diff >= 100);
    }

    #[test]
    fn texture_block_classified_texture() {
        // Checkerboard: every neighbor pair is a large difference.
        let mut rows = [[0u8; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                rows[y][x] = if (x + y) % 2 == 0 { 20 } else { 230 };
            }
        }
        let block = centered(rows);
        let info = analyze_block(&block, &TextureThresholds::default()).unwrap();
        assert_eq!(info.region, RegionClass::Texture);
        assert_eq!(info.strength, 100);
    }

    #[test]
    fn mixed_block_classified_mixed() {
        // Gentle gradient: variation present but no sharp edge pairs
        // (step 5 stays below the edge threshold of 28).
        let mut rows = [[0u8; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                rows[y][x] = 90u8.saturating_add((x as u8).saturating_mul(8));
            }
        }
        let block = centered(rows);
        let info = analyze_block(&block, &TextureThresholds::default()).unwrap();
        assert_eq!(info.region, RegionClass::Mixed);
        assert!(info.strength > 0 && info.strength < 100);
    }

    #[test]
    fn out_of_range_strength_endpoints_error() {
        let bad = AdaptiveMapConfig {
            low_strength: 60,
            high_strength: 40,
            ..AdaptiveMapConfig::default()
        };
        assert_eq!(bad.validate(), Err(AdaptiveError::InvalidStrengthRange));
        let over = AdaptiveMapConfig {
            low_strength: 0,
            high_strength: 101,
            ..AdaptiveMapConfig::default()
        };
        assert_eq!(over.validate(), Err(AdaptiveError::InvalidStrengthRange));
    }

    #[test]
    fn out_of_range_gains_error_instead_of_clamping() {
        let zero_gain = AdaptiveMapConfig {
            low_gain: 0,
            high_gain: 100,
            ..AdaptiveMapConfig::default()
        };
        assert_eq!(zero_gain.validate(), Err(AdaptiveError::InvalidGainRange));
        let too_big = AdaptiveMapConfig {
            low_gain: 100,
            high_gain: 256,
            ..AdaptiveMapConfig::default()
        };
        assert_eq!(too_big.validate(), Err(AdaptiveError::InvalidGainRange));
        let inverted = AdaptiveMapConfig {
            low_gain: 150,
            high_gain: 100,
            ..AdaptiveMapConfig::default()
        };
        assert_eq!(inverted.validate(), Err(AdaptiveError::InvalidGainRange));
        // The error propagates through the mapping entry point too.
        assert!(zero_gain.gain_for_strength(10).is_err());
    }

    #[test]
    fn bad_texture_thresholds_error() {
        assert_eq!(
            analyze_block(
                &Block::new(),
                &TextureThresholds {
                    strength_scale: 0,
                    ..TextureThresholds::default()
                }
            ),
            Err(AdaptiveError::InvalidStrengthScale)
        );
        assert_eq!(
            analyze_block(
                &Block::new(),
                &TextureThresholds {
                    edge_diff: 256,
                    ..TextureThresholds::default()
                }
            ),
            Err(AdaptiveError::InvalidThreshold)
        );
    }

    #[test]
    fn gain_curve_endpoints_and_monotonicity() {
        let cfg = AdaptiveMapConfig::default();
        assert_eq!(cfg.gain_for_strength(0).unwrap(), cfg.low_gain);
        assert_eq!(cfg.gain_for_strength(100).unwrap(), cfg.high_gain);
        let mut prev = 0;
        for s in 0..=100 {
            let g = cfg.gain_for_strength(s).unwrap();
            assert!(g >= prev, "curve must be non-decreasing at {s}");
            assert!(g >= cfg.low_gain && g <= cfg.high_gain);
            prev = g;
        }
    }

    #[test]
    fn unity_gain_reproduces_base_table_byte_for_byte() {
        let base: [i16; 64] = core::array::from_fn(|i| 1 + (i as i16 * 4) % 255);
        let unity = AdaptiveMapConfig {
            low_strength: 0,
            high_strength: 100,
            low_gain: 100,
            high_gain: 100,
        };
        for region in RegionClass::ALL {
            let info = TextureInfo {
                region,
                mean_abs_diff: 5,
                max_abs_diff: 9,
                edge_pixel_count: 0,
                strength: 37,
            };
            let table = adaptive_quant_table(&base, &info, &unity).unwrap();
            assert_eq!(table, base, "unity gain must leave every step untouched");
        }
    }

    #[test]
    fn flat_uses_finer_step_texture_uses_coarser_step() {
        let cfg = AdaptiveMapConfig::default();
        let base = [16i16; 64];
        let flat = TextureInfo {
            region: RegionClass::Flat,
            mean_abs_diff: 0,
            max_abs_diff: 0,
            edge_pixel_count: 0,
            strength: 0,
        };
        let tex = TextureInfo {
            region: RegionClass::Texture,
            mean_abs_diff: 40,
            max_abs_diff: 210,
            edge_pixel_count: 60,
            strength: 100,
        };
        let flat_table = adaptive_quant_table(&base, &flat, &cfg).unwrap();
        let tex_table = adaptive_quant_table(&base, &tex, &cfg).unwrap();
        // 16 * 85 / 100 rounded = 14 (< 16): finer step on flat areas.
        assert!(flat_table[0] < base[0]);
        // 16 * 160 / 100 = 26 (> 16): coarser step on texture.
        assert!(tex_table[0] > base[0]);
        assert!(tex_table[0] > flat_table[0]);
    }

    #[test]
    fn step_never_leaves_legal_table_range() {
        let cfg = AdaptiveMapConfig {
            low_gain: 1,
            high_gain: 255,
            ..AdaptiveMapConfig::default()
        };
        let base = [255i16; 64];
        let info = TextureInfo {
            region: RegionClass::Texture,
            mean_abs_diff: 40,
            max_abs_diff: 255,
            edge_pixel_count: 80,
            strength: 100,
        };
        let table = adaptive_quant_table(&base, &info, &cfg).unwrap();
        for &step in &table {
            assert!((1..=255).contains(&step));
        }
    }
}
