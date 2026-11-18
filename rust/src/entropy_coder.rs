//! Pluggable entropy-coding interface.
//!
//! The encoder main flow (DCT → quant → coefficient walk) no longer writes
//! bits itself. It only drives a backend through three calls:
//!
//!   1. `init()`          — begin one plane (reset state, reserve framing)
//!   2. `encode_symbol()` — encode one symbol (DC category / AC run-size /
//!      raw magnitude bits via `encode_raw_bits`)
//!   3. `finish()`        — seal the plane and return its bytes
//!
//! Backends live in their own modules and are interchangeable:
//!   - `huffman::HuffmanEncoder` — static JPEG Annex K tables (default;
//!     byte-identical to the historical hardcoded path)
//!   - `arith::ArithEncoder`     — zero-order adaptive arithmetic coder
//!
//! A second arithmetic revision (context modeling, etc.) can be added later
//! by implementing the same trait — the walk below does not change.
//!
//! The interface moves symbols by value and each backend returns the `Vec`
//! it built in place: no buffer is copied across the interface boundary.

use crate::block::Block;
use crate::zigzag::ZIGZAG;

/// Symbol classes the coefficient walk can emit. Each class is an independent
/// alphabet; adaptive backends keep one probability model per class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SymClass {
    /// DC magnitude category, luma planes (Y, A). Alphabet: 0..=11.
    DcLuma,
    /// DC magnitude category, chroma planes (Cb, Cr). Alphabet: 0..=11.
    DcChroma,
    /// AC run/size symbol (JPEG RS byte), luma tables. Alphabet: 0..=255.
    AcLuma,
    /// AC run/size symbol, chroma tables. Alphabet: 0..=255.
    AcChroma,
}

/// Encoder backend: init → encode_symbol/encode_raw_bits → finish.
pub trait EntropyEncoder {
    /// Reset all state and prepare the output buffer for one new plane.
    fn init(&mut self);
    /// Encode one symbol of the given class.
    fn encode_symbol(&mut self, class: SymClass, sym: u8);
    /// Encode `n` raw magnitude bits (MSB first), unmodeled.
    fn encode_raw_bits(&mut self, bits: u16, n: u8);
    /// Seal the plane. Returns `[len: u32 LE][payload]` so the decoder can
    /// skip exactly to the next plane boundary.
    fn finish(&mut self) -> Vec<u8>;
}

/// Decoder backend: the mirror of [`EntropyEncoder`]. One `decode_*` call per
/// `encode_*` call, in the same order, reproduces the symbol stream.
pub trait EntropyDecoder {
    /// Decode one symbol of the given class.
    fn decode_symbol(&mut self, class: SymClass) -> Option<u8>;
    /// Decode `n` raw magnitude bits (MSB first).
    fn decode_raw_bits(&mut self, n: u8) -> Option<u16>;
}

// ---------------------------------------------------------------------------
// Magnitude helpers (shared by all backends)
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn category(v: i16) -> u8 {
    if v == 0 { return 0; }
    (16 - v.unsigned_abs().leading_zeros()) as u8
}

#[inline]
pub(crate) fn magnitude_bits(v: i16, cat: u8) -> u16 {
    if v >= 0 { v as u16 } else { ((1u16 << cat) - 1).wrapping_add(v as u16) }
}

#[inline]
pub(crate) fn magnitude_decode(bits: u16, cat: u8) -> i16 {
    let threshold = 1u16 << (cat - 1);
    if bits >= threshold { bits as i16 } else { bits as i16 - (1i16 << cat) + 1 }
}

// ---------------------------------------------------------------------------
// Coefficient walk — the only place the encoder touches entropy coding
// ---------------------------------------------------------------------------

/// Walk all blocks of a plane and emit their symbols through `enc`.
/// Emission order is exactly the historical hardcoded Huffman order:
/// per block: DC category (+bits), then AC run/size symbols (+bits) up to
/// the last non-zero coefficient, then EOB.
///
/// Dispatches to a monomorphized walk per (is_chroma, use_chroma_ac) so the
/// symbol class is a compile-time constant inside the loop — the backend's
/// `encode_symbol` sees no per-symbol class branch.
pub fn encode_plane_walk<E: EntropyEncoder + ?Sized>(
    enc: &mut E,
    blocks: &[Block],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) {
    match (is_chroma, use_chroma_ac) {
        (false, false) => encode_plane_walk_impl::<E, false, false>(enc, blocks, use_dc_delta),
        (false, true) => encode_plane_walk_impl::<E, false, true>(enc, blocks, use_dc_delta),
        (true, false) => encode_plane_walk_impl::<E, true, false>(enc, blocks, use_dc_delta),
        (true, true) => encode_plane_walk_impl::<E, true, true>(enc, blocks, use_dc_delta),
    }
}

fn encode_plane_walk_impl<E: EntropyEncoder + ?Sized, const IS_CHROMA: bool, const CHROMA_AC: bool>(
    enc: &mut E,
    blocks: &[Block],
    use_dc_delta: bool,
) {
    let dc_class = if IS_CHROMA { SymClass::DcChroma } else { SymClass::DcLuma };
    let ac_class = if CHROMA_AC { SymClass::AcChroma } else { SymClass::AcLuma };
    let mut prev_dc: i16 = 0;

    for block in blocks {
        // DC
        let dc_val = block.data[ZIGZAG[0]];
        let dc_emit = if use_dc_delta {
            let d = dc_val.wrapping_sub(prev_dc);
            prev_dc = dc_val;
            d
        } else {
            dc_val
        };
        let dc_cat = category(dc_emit);
        enc.encode_symbol(dc_class, dc_cat);
        if dc_cat > 0 {
            enc.encode_raw_bits(magnitude_bits(dc_emit, dc_cat), dc_cat);
        }

        // AC: stop at last non-zero, emit EOB only when needed.
        let mut last_nz = 63usize;
        while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
            last_nz -= 1;
        }
        if last_nz == 0 {
            enc.encode_symbol(ac_class, 0x00); // EOB
            continue;
        }

        let mut zero_run: u8 = 0;
        for i in 1..=last_nz {
            let val = block.data[ZIGZAG[i]];
            if val == 0 {
                zero_run += 1;
                continue;
            }
            while zero_run >= 16 {
                enc.encode_symbol(ac_class, 0xF0); // ZRL
                zero_run -= 16;
            }
            let cat = category(val);
            let sym = (zero_run << 4) | cat;
            enc.encode_symbol(ac_class, sym);
            enc.encode_raw_bits(magnitude_bits(val, cat), cat);
            zero_run = 0;
        }
        enc.encode_symbol(ac_class, 0x00); // EOB
    }
}

/// Mirror of [`encode_plane_walk`]: decode `n_blocks` blocks from `dec`.
/// Used by backends that decode through the interface (arithmetic path).
pub fn decode_plane_walk<D: EntropyDecoder + ?Sized>(
    dec: &mut D,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Option<Vec<Block>> {
    match (is_chroma, use_chroma_ac) {
        (false, false) => decode_plane_walk_impl::<D, false, false>(dec, n_blocks, use_dc_delta),
        (false, true) => decode_plane_walk_impl::<D, false, true>(dec, n_blocks, use_dc_delta),
        (true, false) => decode_plane_walk_impl::<D, true, false>(dec, n_blocks, use_dc_delta),
        (true, true) => decode_plane_walk_impl::<D, true, true>(dec, n_blocks, use_dc_delta),
    }
}

fn decode_plane_walk_impl<D: EntropyDecoder + ?Sized, const IS_CHROMA: bool, const CHROMA_AC: bool>(
    dec: &mut D,
    n_blocks: usize,
    use_dc_delta: bool,
) -> Option<Vec<Block>> {
    let dc_class = if IS_CHROMA { SymClass::DcChroma } else { SymClass::DcLuma };
    let ac_class = if CHROMA_AC { SymClass::AcChroma } else { SymClass::AcLuma };

    let mut blocks = Vec::with_capacity(n_blocks);
    let mut prev_dc: i16 = 0;
    for _ in 0..n_blocks {
        let mut block = Block::new();

        // DC
        let dc_cat = dec.decode_symbol(dc_class)?;
        if dc_cat > 11 {
            return None;
        }
        let dc_diff = if dc_cat == 0 {
            0i16
        } else {
            magnitude_decode(dec.decode_raw_bits(dc_cat)?, dc_cat)
        };
        let dc_val = if use_dc_delta {
            let v = prev_dc.wrapping_add(dc_diff);
            prev_dc = v;
            v
        } else {
            dc_diff
        };
        block.data[ZIGZAG[0]] = dc_val;

        // AC
        let mut ac_idx = 1usize;
        loop {
            let sym = dec.decode_symbol(ac_class)?;
            if sym == 0x00 {
                break; // EOB
            }
            if sym == 0xF0 {
                // ZRL: exactly 16 consecutive zeros.
                if ac_idx + 16 > 64 {
                    return None;
                }
                ac_idx += 16;
                continue;
            }
            let run = (sym >> 4) as usize;
            let cat = sym & 0x0F;
            if cat == 0 || cat > 10 {
                return None;
            }
            if ac_idx + run >= 64 {
                return None;
            }
            ac_idx += run;
            let bits = dec.decode_raw_bits(cat)?;
            block.data[ZIGZAG[ac_idx]] = magnitude_decode(bits, cat);
            ac_idx += 1;
            if ac_idx > 64 {
                return None;
            }
        }

        blocks.push(block);
    }

    Some(blocks)
}
