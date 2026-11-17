//! Temporal-spatial video denoise for noisy burst sequences.
//!
//! Three stages per frame:
//! 1. Block-level integer motion estimation (full search, configurable range).
//! 2. Motion-adaptive temporal filter: matched blocks are blended with the
//!    recursive previous output; large motion / failed matches blend less,
//!    or not at all. Frame-level fallback when too many blocks fail.
//! 3. Spatial edge-preserving (piecewise-linear bilateral) cleanup pass.
//!
//! Design constraints honored here:
//! * std-only, no third-party crates and no network access.
//! * All scratch storage is allocated once in [`Denoiser::new`]; per-frame
//!   processing performs no heap allocation (see [`Denoiser::buffer_bytes`]).
//! * Pure integer/fixed-point arithmetic -> deterministic, bit-reproducible.
//! * This module is independent of the `.bg` encoder/decoder path; enabling it
//!   cannot change intra-frame encoded bytes (a golden-byte test guards that).

/// Tunable parameters. Every knob mentioned in the task is exposed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenoiseConfig {
    /// Motion-estimation block edge in pixels (1..=min(width, height)).
    pub block_size: usize,
    /// Full-search radius in pixels: candidates cover +/- search_range on
    /// both axes (0 => zero-motion candidate only, always falls back).
    pub search_range: usize,
    /// Temporal blend strength at zero motion, 0..=100 (percent weight).
    pub temporal_strength: u8,
    /// Spatial bilateral range strength, 0..=100 (0 disables the pass).
    pub spatial_strength: u8,
    /// Mean absolute difference per pixel above which a block match is
    /// declared a motion-estimation failure.
    pub match_threshold: u32,
    /// Percentage of failed blocks (0..=100) above which the whole frame
    /// skips temporal filtering (frame-level fallback).
    pub max_failure_ratio: u8,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            block_size: 8,
            search_range: 4,
            temporal_strength: 60,
            spatial_strength: 50,
            match_threshold: 30,
            max_failure_ratio: 50,
        }
    }
}

/// Q8 fixed-point unity.
const Q8: u32 = 256;
/// Pixel-difference knee for the per-pixel occlusion gate (in levels).
const PIXEL_DIFF_KNEE: u32 = 64;
/// Pixel differences at or below this keep the full block weight.
const PIXEL_DIFF_FLOOR: u32 = 16;

/// Streaming denoiser with a fixed, frame-count-independent memory footprint.
pub struct Denoiser {
    width: usize,
    height: usize,
    config: DenoiseConfig,
    blocks_x: usize,
    blocks_y: usize,
    // Four fixed-size pixel planes (N = width*height each).
    cur: Vec<u8>,      // current input frame
    tmp: Vec<u8>,      // temporal-stage output
    prev_in: Vec<u8>,  // previous raw input (ME reference + change gate)
    prev_out: Vec<u8>, // previous final output (recursive blend source)
    // Per-block motion data.
    mvx: Vec<i16>,
    mvy: Vec<i16>,
    sad: Vec<u32>,
    failed: Vec<u8>,
    have_prev: bool,
    frame_fell_back: bool,
    last_failure_ratio: f32,
}

impl Denoiser {
    /// Allocate all fixed frame buffers once. Returns `None` on invalid
    /// parameters or dimensions. No allocation happens afterwards.
    pub fn new(width: usize, height: usize, config: DenoiseConfig) -> Option<Self> {
        if width == 0 || height == 0 || width.checked_mul(height).is_none() {
            return None;
        }
        if config.block_size == 0
            || config.block_size > width
            || config.block_size > height
            || config.temporal_strength > 100
            || config.spatial_strength > 100
            || config.max_failure_ratio > 100
        {
            return None;
        }
        let n = width * height;
        let blocks_x = width.div_ceil(config.block_size);
        let blocks_y = height.div_ceil(config.block_size);
        let b = blocks_x * blocks_y;
        Some(Self {
            width,
            height,
            config,
            blocks_x,
            blocks_y,
            cur: vec![0u8; n],
            tmp: vec![0u8; n],
            prev_in: vec![0u8; n],
            prev_out: vec![0u8; n],
            mvx: vec![0i16; b],
            mvy: vec![0i16; b],
            sad: vec![0u32; b],
            failed: vec![0u8; b],
            have_prev: false,
            frame_fell_back: false,
            last_failure_ratio: 1.0,
        })
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn config(&self) -> DenoiseConfig {
        self.config
    }

    /// Total heap footprint of the internal buffers in bytes. This value is
    /// fixed at construction and never grows with the number of frames
    /// processed (four pixel planes plus small per-block motion tables).
    pub fn buffer_bytes(&self) -> usize {
        let n = self.width * self.height;
        let b = self.blocks_x * self.blocks_y;
        4 * n + 2 * b * size_of::<i16>() + b * size_of::<u32>() + b
    }

    /// Whether the most recently processed frame used the frame-level
    /// temporal fallback (first frame, or too many failed block matches).
    pub fn last_frame_fell_back(&self) -> bool {
        self.frame_fell_back
    }

    /// Fraction of blocks (0.0..=1.0) whose motion estimate failed on the
    /// most recently processed frame.
    pub fn last_failure_ratio(&self) -> f32 {
        self.last_failure_ratio
    }

    /// Convenience wrapper that allocates the output plane. For a strictly
    /// zero-allocation steady state, use [`Denoiser::process_frame`].
    pub fn process_vec(&mut self, frame: &[u8]) -> Option<Vec<u8>> {
        if frame.len() != self.width * self.height {
            return None;
        }
        let mut out = vec![0u8; self.width * self.height];
        self.process_frame(frame, &mut out).then_some(out)
    }

    /// Process one grayscale (8-bit planar) frame into `out`.
    /// Returns `false` (without touching `out`) on size mismatch.
    /// No heap allocation is performed by this call.
    pub fn process_frame(&mut self, frame: &[u8], out: &mut [u8]) -> bool {
        let n = self.width * self.height;
        if frame.len() != n || out.len() != n {
            return false;
        }
        self.cur.copy_from_slice(frame);

        if !self.have_prev {
            // First frame: no temporal reference available.
            self.tmp.copy_from_slice(&self.cur);
            self.frame_fell_back = true;
            self.last_failure_ratio = 1.0;
            for f in self.failed.iter_mut() {
                *f = 1;
            }
        } else {
            self.estimate_motion();
            let failed_count = self.failed.iter().filter(|&&f| f != 0).count();
            let total = self.failed.len();
            let ratio = failed_count as f32 / total as f32;
            self.last_failure_ratio = ratio;
            self.frame_fell_back =
                ratio * 100.0 > self.config.max_failure_ratio as f32;
            if self.frame_fell_back {
                // Frame-level fallback: trust the current frame entirely.
                self.tmp.copy_from_slice(&self.cur);
            } else {
                self.temporal_filter();
            }
        }

        self.spatial_filter(out);

        // Rotate fixed buffers for the next frame.
        self.prev_in.copy_from_slice(&self.cur);
        self.prev_out.copy_from_slice(out);
        self.have_prev = true;
        true
    }

    /// Stage 1: block-level full-search integer motion estimation.
    fn estimate_motion(&mut self) {
        let (w, h) = (self.width, self.height);
        let bs = self.config.block_size;
        let r = self.config.search_range;
        let threshold = self.config.match_threshold;
        for by_idx in 0..self.blocks_y {
            for bx_idx in 0..self.blocks_x {
                let block_id = by_idx * self.blocks_x + bx_idx;
                let ox = bx_idx * bs;
                let oy = by_idx * bs;
                let bw = bs.min(w - ox);
                let bh = bs.min(h - oy);
                let count = (bw * bh) as u32;

                let min_dx = -(r.min(ox) as i32);
                let max_dx = r.min(w - (ox + bw)) as i32;
                let min_dy = -(r.min(oy) as i32);
                let max_dy = r.min(h - (oy + bh)) as i32;

                let mut best_sad = u32::MAX;
                let mut best_dx = 0i32;
                let mut best_dy = 0i32;
                for dy in min_dy..=max_dy {
                    for dx in min_dx..=max_dx {
                        let mut sad = 0u32;
                        for yy in 0..bh {
                            let row_cur = (oy + yy) * w + ox;
                            let row_ref = (oy + yy + dy as usize) * w
                                + (ox + dx as usize);
                            for xx in 0..bw {
                                let a = self.cur[row_cur + xx] as i16;
                                let b = self.prev_in[row_ref + xx] as i16;
                                sad = sad.saturating_add(a.abs_diff(b) as u32);
                            }
                        }
                        if sad < best_sad {
                            best_sad = sad;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }

                // Compare raw sums so sub-unit per-pixel differences are not
                // truncated away; `match_threshold` stays a per-pixel knob.
                let fails = best_sad > threshold.saturating_mul(count);
                self.mvx[block_id] = best_dx as i16;
                self.mvy[block_id] = best_dy as i16;
                self.sad[block_id] = best_sad;
                self.failed[block_id] = fails as u8;
            }
        }
    }

    /// Stage 2: motion-adaptive recursive temporal blend.
    fn temporal_filter(&mut self) {
        let (w, h) = (self.width, self.height);
        let bs = self.config.block_size;
        let r = self.config.search_range.max(1) as u32;
        let ts = self.config.temporal_strength as u32; // 0..=100

        for by_idx in 0..self.blocks_y {
            for bx_idx in 0..self.blocks_x {
                let block_id = by_idx * self.blocks_x + bx_idx;
                let ox = bx_idx * bs;
                let oy = by_idx * bs;
                let bw = bs.min(w - ox);
                let bh = bs.min(h - oy);

                // Block weight (Q8): zero on failure, otherwise attenuated by
                // the motion-vector magnitude relative to the search radius.
                let block_w = if self.failed[block_id] != 0 {
                    0
                } else {
                    let dx = self.mvx[block_id].unsigned_abs() as u32;
                    let dy = self.mvy[block_id].unsigned_abs() as u32;
                    let motion = dx.max(dy).min(r);
                    let motion_factor = Q8 - Q8 * motion / r; // Q8
                    ts * motion_factor / 100
                };

                let dx = self.mvx[block_id] as isize;
                let dy = self.mvy[block_id] as isize;

                for yy in 0..bh {
                    for xx in 0..bw {
                        let x = ox + xx;
                        let y = oy + yy;
                        let idx = y * w + x;
                        if block_w == 0 {
                            self.tmp[idx] = self.cur[idx];
                            continue;
                        }
                        let rx = (x as isize + dx) as usize;
                        let ry = (y as isize + dy) as usize;
                        let ridx = ry * w + rx;

                        // Per-pixel occlusion/change gate against the raw
                        // previous input: residual large differences suppress
                        // blending even inside a nominally matched block.
                        let diff = (self.cur[idx] as i16)
                            .abs_diff(self.prev_in[ridx] as i16) as u32;
                        let pixel_factor = if diff >= PIXEL_DIFF_KNEE {
                            0
                        } else if diff <= PIXEL_DIFF_FLOOR {
                            Q8
                        } else {
                            Q8 * (PIXEL_DIFF_KNEE - diff)
                                / (PIXEL_DIFF_KNEE - PIXEL_DIFF_FLOOR)
                        };
                        let weight = block_w * pixel_factor / Q8;
                        if weight == 0 {
                            self.tmp[idx] = self.cur[idx];
                            continue;
                        }
                        let cur_v = self.cur[idx] as u32;
                        let ref_v = self.prev_out[ridx] as u32;
                        let blended =
                            (cur_v * (Q8 - weight) + ref_v * weight + Q8 / 2) / Q8;
                        self.tmp[idx] = blended as u8;
                    }
                }
            }
        }
    }

    /// Stage 3: 3x3 edge-preserving bilateral filter with a piecewise-linear
    /// range kernel and a flat spatial kernel. `strength == 0` copies through.
    fn spatial_filter(&self, out: &mut [u8]) {
        if self.config.spatial_strength == 0 {
            out.copy_from_slice(&self.tmp);
            return;
        }
        let (w, h) = (self.width, self.height);
        // Range knee: 16 levels at low strength, 128 at full strength.
        let knee = (16
            + (128 - 16) * self.config.spatial_strength as u32 / 100)
            .max(1);
        let src = &self.tmp;
        for y in 0..h {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(h - 1);
            for x in 0..w {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(w - 1);
                let idx = y * w + x;
                let center = src[idx] as u32;

                let mut wsum = Q8; // center weight
                let mut acc = center * Q8;
                for &ny in &[ym, yp] {
                    for &nx in &[xm, x, xp] {
                        let nidx = ny * w + nx;
                        let v = src[nidx] as u32;
                        let d = v.abs_diff(center);
                        let rw = if d >= knee {
                            0
                        } else {
                            Q8 * (knee - d) / knee
                        };
                        acc += v * rw;
                        wsum += rw;
                    }
                }
                // Also the two horizontal neighbors on the center row.
                for &nx in &[xm, xp] {
                    let nidx = y * w + nx;
                    let v = src[nidx] as u32;
                    let d = v.abs_diff(center);
                    let rw = if d >= knee {
                        0
                    } else {
                        Q8 * (knee - d) / knee
                    };
                    acc += v * rw;
                    wsum += rw;
                }
                out[idx] = ((acc + wsum / 2) / wsum) as u8;
            }
        }
    }
}

/// Peak signal-to-noise ratio in dB between two equal-length 8-bit planes.
/// Returns [`f32::INFINITY`] for identical planes, `None` on length mismatch.
pub fn psnr(reference: &[u8], distorted: &[u8]) -> Option<f32> {
    if reference.len() != distorted.len() || reference.is_empty() {
        return None;
    }
    let mut se = 0u64;
    for (a, b) in reference.iter().zip(distorted.iter()) {
        let d = *a as i32 - *b as i32;
        se += (d * d) as u64;
    }
    if se == 0 {
        return Some(f32::INFINITY);
    }
    let mse = se as f64 / reference.len() as f64;
    Some(10.0 * (255.0_f64 * 255.0 / mse).log10() as f32)
}

/// Mean absolute error between two equal-length 8-bit planes.
pub fn mae(a: &[u8], b: &[u8]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let sum: u64 = a.iter().zip(b.iter()).map(|(x, y)| x.abs_diff(*y) as u64).sum();
    Some(sum as f32 / a.len() as f32)
}

// ===========================================================================
// Tests + four-sequence evaluation harness (small synthetic data, std-only).
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const W: usize = 64;
    const H: usize = 64;
    const FRAMES: usize = 8;
    const N: usize = W * H;

    /// Deterministic PCG-XSH-RR so every run is reproducible without
    /// external data. A plain LCG is unusable here: with 3 draws per pixel
    /// and 4096 pixels/frame, 12288 draws divide its 2^32 period evenly, so
    /// the noise stream would repeat verbatim on every frame.
    struct Pcg(u64);
    impl Pcg {
        fn new(seed: u32) -> Self {
            let mut s = Self(0xcafef00dd15ea5e5);
            s.next_u32();
            s.0 = s.0.wrapping_add(seed as u64 | 1);
            s.next_u32();
            s
        }
        fn next_u32(&mut self) -> u32 {
            let old = self.0;
            self.0 = old
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
            let rot = (old >> 59) as u32;
            xorshifted.rotate_right(rot)
        }
        /// 3-tap Irwin-Hall, integer-scaled to the requested sigma.
        /// Empirical std at sigma=18 is ~17.6.
        fn gaussian(&mut self, sigma: i32) -> i32 {
            let s = (self.next_u32() % 1024) as i32
                + (self.next_u32() % 1024) as i32
                + (self.next_u32() % 1024) as i32;
            (s - 1536) * sigma / 512
        }
    }

    fn pattern(x: usize, y: usize) -> u8 {
        ((x as u32).wrapping_mul(3).wrapping_add((y as u32).wrapping_mul(2)) & 255) as u8
    }

    #[derive(Clone, Copy)]
    enum Seq {
        Static,
        Slow,
        Fast,
        Occlusion,
    }

    /// Build a clean sequence and its noisy observation (gaussian noise sigma 18).
    fn build_sequence(kind: Seq) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let shift = match kind {
            Seq::Static | Seq::Occlusion => 1usize,
            Seq::Slow => 1,
            Seq::Fast => 6,
        };
        let mut rng = Pcg::new(match kind {
            Seq::Static => 0x1234_5678,
            Seq::Slow => 0x9abc_def0,
            Seq::Fast => 0x0bad_f00d,
            Seq::Occlusion => 0x55aa_5a5a,
        });
        let mut clean = Vec::with_capacity(FRAMES);
        let mut noisy = Vec::with_capacity(FRAMES);
        for f in 0..FRAMES {
            let mut base = vec![0u8; N];
            for y in 0..H {
                for x in 0..W {
                    let sx = match kind {
                        Seq::Static => x,
                        _ => (x as i32 - (shift * f) as i32).rem_euclid(W as i32) as usize,
                    };
                    base[y * W + x] = pattern(sx, y);
                }
            }
            if matches!(kind, Seq::Occlusion) {
                // Solid block traversing the frame: covers/uncovers content.
                let x0 = (4 + 5 * f) % W;
                for y in 20..34 {
                    for k in 0..14 {
                        let x = (x0 + k) % W;
                        base[y * W + x] = 230;
                    }
                }
            }
            let mut obs = vec![0u8; N];
            for i in 0..N {
                let g = rng.gaussian(18);
                let v = base[i] as i32 + g;
                obs[i] = v.clamp(0, 255) as u8;
            }
            clean.push(base);
            noisy.push(obs);
        }
        (clean, noisy)
    }

    #[test]
    fn invalid_configs_rejected() {
        let c = DenoiseConfig::default();
        assert!(Denoiser::new(0, 8, c).is_none());
        assert!(Denoiser::new(8, 8, DenoiseConfig { block_size: 0, ..c }).is_none());
        assert!(Denoiser::new(8, 8, DenoiseConfig { block_size: 16, ..c }).is_none());
        assert!(Denoiser::new(8, 8, DenoiseConfig { temporal_strength: 101, ..c }).is_none());
        assert!(Denoiser::new(8, 8, DenoiseConfig { spatial_strength: 200, ..c }).is_none());
        assert!(Denoiser::new(8, 8, DenoiseConfig { max_failure_ratio: 101, ..c }).is_none());
    }

    #[test]
    fn psnr_and_mae_basics() {
        assert_eq!(psnr(&[10u8, 20], &[10, 20]), Some(f32::INFINITY));
        assert!(psnr(&[0u8, 0], &[3, 4]).is_some());
        // mse = (9+16)/2 = 12.5
        let expected = 10.0 * (65025.0_f32 / 12.5).log10();
        assert!((psnr(&[0u8, 0], &[3, 4]).unwrap() - expected).abs() < 1e-4);
        assert_eq!(mae(&[0u8, 10], &[4, 20]), Some(7.0));
        assert_eq!(mae(&[0u8], &[1, 2]), None);
    }

    #[test]
    fn first_frame_is_passthrough_when_spatial_disabled() {
        let cfg = DenoiseConfig { spatial_strength: 0, ..Default::default() };
        let mut d = Denoiser::new(W, H, cfg).unwrap();
        let (_, noisy) = build_sequence(Seq::Static);
        let out = d.process_vec(&noisy[0]).unwrap();
        assert_eq!(out, noisy[0]);
        assert!(d.last_frame_fell_back());
        assert_eq!(d.last_failure_ratio(), 1.0);
    }

    #[test]
    fn frame_level_fallback_on_impossible_threshold() {
        // Any nonzero per-pixel difference fails the match; independent noise
        // on consecutive frames makes some blocks fail, and a tight
        // max_failure_ratio turns that into the frame-level fallback.
        let cfg = DenoiseConfig {
            match_threshold: 0,
            max_failure_ratio: 20,
            spatial_strength: 0,
            ..Default::default()
        };
        let mut d = Denoiser::new(W, H, cfg).unwrap();
        let (_, noisy) = build_sequence(Seq::Static);
        d.process_vec(&noisy[0]).unwrap();
        let out = d.process_vec(&noisy[1]).unwrap();
        assert!(d.last_frame_fell_back());
        // With temporal and spatial both bypassed, output must equal input.
        assert_eq!(out, noisy[1]);
    }

    #[test]
    fn zero_search_range_keeps_only_zero_motion_candidate() {
        // search_range 0 leaves only the (0,0) candidate. On a fast-moving
        // sequence it cannot track the shift, so with the strictest match
        // threshold every block fails and the frame-level fallback engages.
        let cfg = DenoiseConfig {
            search_range: 0,
            match_threshold: 0,
            spatial_strength: 0,
            ..Default::default()
        };
        let mut d = Denoiser::new(W, H, cfg).unwrap();
        let (_, noisy) = build_sequence(Seq::Fast);
        d.process_vec(&noisy[0]).unwrap();
        let out = d.process_vec(&noisy[1]).unwrap();
        assert_eq!(d.last_failure_ratio(), 1.0);
        assert!(d.last_frame_fell_back());
        assert_eq!(out, noisy[1]);
    }

    #[test]
    fn output_is_deterministic() {
        let (_, noisy) = build_sequence(Seq::Slow);
        let run = || {
            let mut d = Denoiser::new(W, H, DenoiseConfig::default()).unwrap();
            let mut outs = Vec::new();
            for f in noisy.iter() {
                outs.push(d.process_vec(f).unwrap());
            }
            outs
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn fixed_buffers_never_grow() {
        let mut d = Denoiser::new(W, H, DenoiseConfig::default()).unwrap();
        let bytes_before = d.buffer_bytes();
        let caps_before: usize = d.cur.capacity()
            + d.tmp.capacity()
            + d.prev_in.capacity()
            + d.prev_out.capacity()
            + d.mvx.capacity() * size_of::<i16>()
            + d.mvy.capacity() * size_of::<i16>()
            + d.sad.capacity() * size_of::<u32>()
            + d.failed.capacity();
        let (_, noisy) = build_sequence(Seq::Occlusion);
        let mut out = vec![0u8; N];
        for f in noisy.iter() {
            assert!(d.process_frame(f, &mut out));
        }
        let caps_after: usize = d.cur.capacity()
            + d.tmp.capacity()
            + d.prev_in.capacity()
            + d.prev_out.capacity()
            + d.mvx.capacity() * size_of::<i16>()
            + d.mvy.capacity() * size_of::<i16>()
            + d.sad.capacity() * size_of::<u32>()
            + d.failed.capacity();
        assert_eq!(caps_before, caps_after);
        assert_eq!(d.buffer_bytes(), bytes_before);
        // Footprint is independent of sequence length by construction.
        assert_eq!(bytes_before, 4 * N + 9 * (W / 8) * (H / 8));
    }

    #[test]
    fn zero_strengths_is_exact_passthrough() {
        // Both filter stages disabled: output must equal input on every frame.
        let cfg = DenoiseConfig {
            temporal_strength: 0,
            spatial_strength: 0,
            ..Default::default()
        };
        let mut d = Denoiser::new(W, H, cfg).unwrap();
        let (_, noisy) = build_sequence(Seq::Slow);
        for frame in noisy.iter() {
            assert_eq!(d.process_vec(frame).unwrap(), *frame);
        }
    }

    #[test]
    fn larger_search_range_reduces_failures_on_fast_motion() {
        let (_, noisy) = build_sequence(Seq::Fast);
        let run = |range: usize| -> f32 {
            let cfg = DenoiseConfig { search_range: range, ..Default::default() };
            let mut d = Denoiser::new(W, H, cfg).unwrap();
            let mut worst = 0.0_f32;
            for f in noisy.iter() {
                d.process_vec(f).unwrap();
                if !d.last_frame_fell_back() {
                    worst = worst.max(d.last_failure_ratio());
                }
            }
            worst
        };
        // Shift is 6 px/frame; range 4 cannot track it, range 8 can.
        assert!(run(8) < run(4), "wider search range should reduce ME failures");
    }

    #[test]
    fn size_mismatch_is_rejected() {
        let mut d = Denoiser::new(W, H, DenoiseConfig::default()).unwrap();
        let mut out = vec![0u8; N];
        assert!(!d.process_frame(&vec![0u8; N - 1], &mut out));
        let mut small_out = vec![0u8; 1];
        assert!(!d.process_frame(&vec![0u8; N], &mut small_out));
    }

    /// Full evaluation over the four required sequences. Small synthetic
    /// frames; prints the metrics table with `--nocapture`.
    #[test]
    fn evaluate_four_sequences() {
        let cases = [
            (Seq::Static, "static"),
            (Seq::Slow, "slow-motion"),
            (Seq::Fast, "fast-motion"),
            (Seq::Occlusion, "occlusion"),
        ];
        let cfg = DenoiseConfig::default();
        eprintln!();
        eprintln!("config: {:?}", cfg);
        eprintln!(
            "{:<12} | {:>9} | {:>9} | {:>8} | {:>14} | {:>10} | {:>8} | {:>9}",
            "sequence", "PSNR noisy", "PSNR den", "gain dB",
            "motionMAE n/d", "us/frame", "maxFail%", "fallbacks"
        );
        for (kind, label) in cases {
            let (clean, noisy) = build_sequence(kind);
            let mut d = Denoiser::new(W, H, cfg).unwrap();
            let mut denoised = Vec::with_capacity(FRAMES);
            let mut fallbacks = 0usize;
            let mut max_fail = 0.0_f32;
            let t0 = Instant::now();
            for f in noisy.iter() {
                let out = d.process_vec(f).unwrap();
                if d.last_frame_fell_back() {
                    fallbacks += 1;
                } else {
                    max_fail = max_fail.max(d.last_failure_ratio());
                }
                denoised.push(out);
            }
            let elapsed_us = t0.elapsed().as_micros();

            let mut pn = 0.0_f32;
            let mut pd = 0.0_f32;
            let mut mn = 0.0_f32;
            let mut md = 0.0_f32;
            let mut mask_pixels = 0usize;
            for f in 0..FRAMES {
                pn += psnr(&clean[f], &noisy[f]).unwrap();
                pd += psnr(&clean[f], &denoised[f]).unwrap();
                if f > 0 {
                    for i in 0..N {
                        if clean[f][i] != clean[f - 1][i] {
                            mn += noisy[f][i].abs_diff(clean[f][i]) as f32;
                            md += denoised[f][i].abs_diff(clean[f][i]) as f32;
                            mask_pixels += 1;
                        }
                    }
                }
            }
            pn /= FRAMES as f32;
            pd /= FRAMES as f32;
            if mask_pixels > 0 {
                mn /= mask_pixels as f32;
                md /= mask_pixels as f32;
            }
            eprintln!(
                "{:<12} | {:>9.3} | {:>9.3} | {:>+8.3} | {:>6.2}/{:<6.2} | {:>10} | {:>8.1} | {:>9}",
                label, pn, pd, pd - pn, mn, md,
                elapsed_us / FRAMES as u128, max_fail * 100.0, fallbacks
            );

            // Hard acceptance checks.
            match kind {
                Seq::Static | Seq::Slow | Seq::Occlusion => {
                    assert!(pd > pn, "{label}: expected positive PSNR gain");
                }
                Seq::Fast => {
                    // Motion adaptation must not make fast motion worse.
                    assert!(pd >= pn - 0.1, "{label}: fast-motion regression");
                }
            }
            if mask_pixels > 0 {
                // Denoising must not increase error on moving-content pixels
                // (motion blur / ghosting control), with a small tolerance.
                assert!(
                    md <= mn * 1.10 + 0.5,
                    "{label}: motion-region MAE worsened {mn} -> {md}"
                );
            }
        }
        eprintln!(
            "fixed frame-buffer footprint: {} bytes (independent of frame count)",
            Denoiser::new(W, H, cfg).unwrap().buffer_bytes()
        );
    }
}
