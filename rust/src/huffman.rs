//! Entropy layer for DCT coefficients.
//!
//! # Current coder: adaptive binary/multi-symbol arithmetic coding
//!
//! The encoder and decoder in this module implement a bit-oriented arithmetic (range) coder
//! driven by per-plane adaptive frequency models. It replaces the historical static JPEG
//! Huffman tables for the coefficient payload. The old Huffman decoder is retained below as a
//! fallback so `.bg` files written by earlier releases still decode (see *Plane payload*).
//!
//! ## Why arithmetic coding
//!
//! Static Huffman spends whole code words per `(run, size)` symbol and cannot adapt to the
//! actual statistics of one image/plane. The models here are updated from the symbols actually
//! coded, with context (zigzag band, zero-run length, previous magnitude category, previous DC
//! magnitude category), so skewed coefficient distributions cost close to their entropy.
//!
//! ## Coefficient tokenisation (lossless, per 8×8 block)
//!
//! For each block, in zigzag order:
//!   1. `last_nz` ∈ 0..63: index of the last non-zero AC coefficient (0 = no AC). One adaptive
//!      64-symbol model, context = bucket of the previous block's `last_nz`.
//!   2. For `zi` in `1..last_nz` code a binary "non-zero" flag with an adaptive model,
//!      context = `band(zi) * 4 + min(zero_run, 3)`, where `band(zi) = (zi - 1) / 8`.
//!      The coefficient at `zi == last_nz` is known non-zero and has no flag.
//!   3. Every non-zero coefficient (including `zi == last_nz`) is coded as:
//!        - magnitude category `c = floor(log2(|v|))` (0 for |v| = 1) with an adaptive
//!          16-symbol model, context = `band(zi) * 4 + min(prev_cat, 3)`;
//!        - the `c` low magnitude bits as raw arithmetic bits (probability exactly 1/2;
//!          the implicit top bit is not coded because it is always 1 for `c >= 1`);
//!        - the sign with one adaptive binary model per plane.
//!   4. DC is coded before the AC scan as a signed value (delta vs. the previous block's DC
//!      when the profile uses DC prediction), with adaptive category/sign models and a category
//!      context bucketed from the previous DC delta.
//!
//! The same driver (`process_block`) is instantiated for the encoder and the decoder; the
//! decoder simply ignores the "known" symbol and uses the value returned by the arithmetic
//! decoder, which makes both sides bit-symmetric by construction.
//!
//! ## Adaptive frequency models and the update period
//!
//! Every model starts uniform (frequency 1 per symbol). For a coded symbol, with update period
//! `P`:
//!   - `P == 0`: the model is **static** — frequencies never change (uniform code).
//!   - `P == 1`: the symbol's frequency is incremented immediately after it is coded.
//!   - `P > 1`: the symbol is counted in a pending bucket; every `P` symbols coded through that
//!     model the pending counts are folded into the frequencies.
//! After folding, if the total exceeds 32768 the table is rescaled with `f[i] = (f[i] + 1) / 2`
//! (keeps every symbol codeable). Encoder and decoder perform the identical update after coding
//! or decoding each symbol.
//!
//! The period is configurable through the environment variable `BITGRAIN_ARITH_UPDATE_PERIOD`
//! (default 1, clamped to 0..127) and is stored in the stream, so decoding does not depend on
//! the environment used for encoding.
//!
//! ## Plane payload (defines the new bitstream)
//!
//! Each plane keeps the historical framing: `[payload_len: u32 LE][payload]`. The payload is:
//!
//! ```text
//! offset  size  field
//! 0       4     magic "ARIT" (0x41 0x52 0x49 0x54)
//! 4       1     codec version = 1
//! 5       1     config: bit0..6 = update period (0 = static, 1..127), bit7 reserved = 0
//! 6       ...   arithmetic-coded bitstream
//! ```
//!
//! A payload without the `ARIT` magic is decoded with the legacy Huffman path below.
//!
//! ### Bitstream and tail flush
//!
//! The arithmetic coder emits bits MSB-first into bytes (no 0xFF stuffing). The final flush
//! emits the disambiguating bit plus the outstanding underflow ("pending") bits, then pads the
//! last partial byte with zero bits. The decoder reads exactly `payload_len` bytes and supplies
//! zero bits past the end, which is the standard arithmetic-coder convention and matches the
//! encoder's zero padding.
//!
//! # Legacy coder
//!
//! The historical path used the standard JPEG Huffman tables (ISO 10918-1 Annex K): luma DC +
//! full luminance AC for Y (and A), chroma DC for Cb/Cr. Its bitstream is packed MSB-first,
//! continuous across all blocks in a plane, with 0xFF byte stuffing and a final pad of 1s. It is
//! kept read-only for compatibility.

use crate::block::Block;
use crate::zigzag::ZIGZAG;

// ---------------------------------------------------------------------------
// Standard JPEG Huffman tables (ISO 10918-1 Annex K)
// Each entry: (code_length_bits, code_value)
// ---------------------------------------------------------------------------

static LUMA_DC_TABLE: &[(u8, u16)] = &[
    (2, 0b00),
    (3, 0b010),
    (3, 0b011),
    (3, 0b100),
    (3, 0b101),
    (3, 0b110),
    (4, 0b1110),
    (5, 0b11110),
    (6, 0b111110),
    (7, 0b1111110),
    (8, 0b11111110),
    (9, 0b111111110),
];

static CHROMA_DC_TABLE: &[(u8, u16)] = &[
    (2, 0b00),
    (2, 0b01),
    (2, 0b10),
    (3, 0b110),
    (4, 0b1110),
    (5, 0b11110),
    (6, 0b111110),
    (7, 0b1111110),
    (8, 0b11111110),
    (9, 0b111111110),
    (10,0b1111111110),
    (11,0b11111111110),
];

/// AC table indexed by symbol byte: high nibble = run (0–15), low nibble = category (1–10).
/// Symbol 0x00 = EOB, 0xF0 = ZRL. Entry (0,0) means "not in table".
type AcTable = [(u8, u16); 256];

use crate::jpeg_luma_ac_ht::JPEG_LUMA_AC_HT;
use std::sync::OnceLock;
const FAST_BITS: u8 = 14;
type FastEntry = (i16, u8); // (sym, bits_len), sym<0 means miss

/// stb_image_write / ISO 10918-1 Annex K luminance AC Huffman (full 256-entry encode table).
fn ac_table_from_stb_ht(ht: &[[u16; 2]; 256]) -> AcTable {
    let mut t = [(0u8, 0u16); 256];
    for i in 0..256 {
        let code = ht[i][0];
        let len = ht[i][1];
        if len > 0 {
            t[i] = (len as u8, code);
        }
    }
    t
}

/// Full JPEG luminance AC table. Used for **every** plane in .bg v4/v5 (Y, Cb, Cr, A) so the
/// on-wire format stays compatible with earlier bitgrain, which always applied the luma AC spec
/// to chroma too (not the separate chrominance AC table).
static JPEG_AC_TABLE: OnceLock<AcTable> = OnceLock::new();
static JPEG_CHROMA_AC_TABLE: OnceLock<AcTable> = OnceLock::new();

fn jpeg_ac_table() -> &'static AcTable {
    JPEG_AC_TABLE.get_or_init(|| ac_table_from_stb_ht(&JPEG_LUMA_AC_HT))
}

fn ac_table_from_canonical(nrcodes: &[u8; 17], values: &[u8]) -> AcTable {
    let mut t = [(0u8, 0u16); 256];
    let mut code: u16 = 0;
    let mut k = 0usize;
    for len in 1..=16usize {
        let n = nrcodes[len] as usize;
        for _ in 0..n {
            let sym = values[k] as usize;
            t[sym] = (len as u8, code);
            code = code.wrapping_add(1);
            k += 1;
        }
        code <<= 1;
    }
    t
}

fn jpeg_chroma_ac_table() -> &'static AcTable {
    JPEG_CHROMA_AC_TABLE.get_or_init(|| {
        // JPEG Annex K / stb_image_write chrominance AC Huffman table.
        const NR_CODES: [u8; 17] = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
        const VALUES: [u8; 162] = [
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
            0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
            0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
            0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
            0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
            0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
            0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
            0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
            0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
            0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
            0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
            0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
            0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
            0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
            0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
            0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa,
        ];
        ac_table_from_canonical(&NR_CODES, &VALUES)
    })
}

#[derive(Clone, Copy)]
struct DecodeNode {
    child: [i16; 2],
    sym: i16,
}

impl DecodeNode {
    #[inline]
    fn new() -> Self {
        Self { child: [-1, -1], sym: -1 }
    }
}

struct DecodeTree {
    nodes: Vec<DecodeNode>,
    fast: [FastEntry; 1 << FAST_BITS],
}

impl DecodeTree {
    fn with_root() -> Self {
        Self {
            nodes: vec![DecodeNode::new()],
            fast: [(-1, 0); 1 << FAST_BITS],
        }
    }

    fn insert(&mut self, code: u16, len: u8, sym: u8) -> bool {
        if len == 0 || len > 16 {
            return false;
        }
        let mut idx = 0usize;
        for shift in (0..len).rev() {
            // A leaf cannot be extended.
            if self.nodes[idx].sym >= 0 {
                return false;
            }
            let bit = ((code >> shift) & 1) as usize;
            let next = self.nodes[idx].child[bit];
            let next_idx = if next >= 0 {
                next as usize
            } else {
                self.nodes.push(DecodeNode::new());
                let ni = (self.nodes.len() - 1) as i16;
                self.nodes[idx].child[bit] = ni;
                ni as usize
            };
            idx = next_idx;
        }
        // A node with children cannot become a leaf; an existing leaf cannot be overwritten.
        if self.nodes[idx].sym >= 0 || self.nodes[idx].child[0] >= 0 || self.nodes[idx].child[1] >= 0 {
            return false;
        }
        self.nodes[idx].sym = sym as i16;

        // Build a small prefix LUT for fast symbol decode.
        if len <= FAST_BITS {
            let shift = (FAST_BITS - len) as usize;
            let base = (code as usize) << shift;
            let reps = 1usize << shift;
            for i in 0..reps {
                self.fast[base + i] = (sym as i16, len);
            }
        }
        true
    }
}

static LUMA_DC_TREE: OnceLock<DecodeTree> = OnceLock::new();
static CHROMA_DC_TREE: OnceLock<DecodeTree> = OnceLock::new();
static AC_TREE_LUMA: OnceLock<DecodeTree> = OnceLock::new();
static AC_TREE_CHROMA: OnceLock<DecodeTree> = OnceLock::new();

fn build_dc_tree(table: &[(u8, u16)]) -> DecodeTree {
    let mut t = DecodeTree::with_root();
    for (sym, &(len, code)) in table.iter().enumerate() {
        assert!(t.insert(code, len, sym as u8), "invalid DC Huffman table");
    }
    t
}

fn build_ac_tree(table: &AcTable) -> DecodeTree {
    let mut t = DecodeTree::with_root();
    for (sym, &(len, code)) in table.iter().enumerate() {
        if len > 0 {
            assert!(t.insert(code, len, sym as u8), "invalid AC Huffman table");
        }
    }
    t
}

#[inline]
fn luma_dc_tree() -> &'static DecodeTree {
    LUMA_DC_TREE.get_or_init(|| build_dc_tree(LUMA_DC_TABLE))
}

#[inline]
fn chroma_dc_tree() -> &'static DecodeTree {
    CHROMA_DC_TREE.get_or_init(|| build_dc_tree(CHROMA_DC_TABLE))
}

#[inline]
fn ac_tree(use_chroma_ac: bool) -> &'static DecodeTree {
    if use_chroma_ac {
        AC_TREE_CHROMA.get_or_init(|| build_ac_tree(jpeg_chroma_ac_table()))
    } else {
        AC_TREE_LUMA.get_or_init(|| build_ac_tree(jpeg_ac_table()))
    }
}

// ---------------------------------------------------------------------------
// Magnitude helpers
// ---------------------------------------------------------------------------

#[inline]
#[allow(dead_code)] // used by the test-only legacy encoder
fn category(v: i16) -> u8 {
    if v == 0 { return 0; }
    (16 - v.unsigned_abs().leading_zeros()) as u8
}

#[inline]
#[allow(dead_code)] // used by the test-only legacy encoder
fn magnitude_bits(v: i16, cat: u8) -> u16 {
    if v >= 0 { v as u16 } else { ((1u16 << cat) - 1).wrapping_add(v as u16) }
}

#[inline]
fn magnitude_decode(bits: u16, cat: u8) -> i16 {
    let threshold = 1u16 << (cat - 1);
    if bits >= threshold { bits as i16 } else { bits as i16 - (1i16 << cat) + 1 }
}

/// JPEG-style DC/AC Huffman tables only cover DC category ≤11 and AC category ≤10.
/// Without this clamp, `encode_plane` can hit the `al == 0` fallback and emit a bitstream
/// the decoder cannot parse (decode returns `None` / CLI reports corrupt .bg).
pub(crate) fn clamp_block_jpeg_coeffs(block: &mut Block) {
    const DC_MAX: i16 = 2047; // category 11 magnitude
    const AC_MAX: i16 = 1023; // category 10 magnitude
    block.data[ZIGZAG[0]] = block.data[ZIGZAG[0]].clamp(-DC_MAX, DC_MAX);
    for i in 1..64 {
        let z = ZIGZAG[i];
        block.data[z] = block.data[z].clamp(-AC_MAX, AC_MAX);
    }
}

// ---------------------------------------------------------------------------
// Bit writer — continuous across blocks, flush once per plane
// ---------------------------------------------------------------------------

pub struct BitWriter {
    pub buf:     Vec<u8>,
    bit_buf:     u64,
    bits_in:     u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self { buf: Vec::with_capacity(4096), bit_buf: 0, bits_in: 0 }
    }

    #[inline]
    fn push_entropy_byte(&mut self, byte: u8) {
        self.buf.push(byte);
        if byte == 0xFF {
            self.buf.push(0x00);
        }
    }

    #[inline]
    pub fn write_bits(&mut self, code: u16, n: u8) {
        if n == 0 { return; }
        let mask = (1u64 << n) - 1;
        self.bit_buf = (self.bit_buf << n) | ((code as u64) & mask);
        self.bits_in += n;

        while self.bits_in >= 8 {
            let shift = self.bits_in - 8;
            let byte = (self.bit_buf >> shift) as u8;
            self.push_entropy_byte(byte);
            self.bits_in -= 8;
            if self.bits_in == 0 {
                self.bit_buf = 0;
            } else {
                self.bit_buf &= (1u64 << self.bits_in) - 1;
            }
        }
    }

    /// Flush remaining bits, padding with 1s (JPEG convention). Call once per plane.
    pub fn flush(&mut self) {
        if self.bits_in > 0 {
            let pad = 8 - self.bits_in;
            self.write_bits((1u16 << pad) - 1, pad);
        }
    }
}

// ---------------------------------------------------------------------------
// Bit reader — continuous across blocks
// ---------------------------------------------------------------------------

pub struct BitReader<'a> {
    buf:     &'a [u8],
    pos:     usize,
    bit_buf: u64,
    bits_in: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8], start: usize) -> Self {
        Self { buf, pos: start, bit_buf: 0, bits_in: 0 }
    }

    #[inline]
    fn refill(&mut self) {
        while self.bits_in <= 56 && self.pos < self.buf.len() {
            let byte = self.buf[self.pos];
            self.pos += 1;
            // JPEG byte-unstuffing
            if byte == 0xFF && self.pos < self.buf.len() && self.buf[self.pos] == 0x00 {
                self.pos += 1;
            }
            self.bit_buf = (self.bit_buf << 8) | byte as u64;
            self.bits_in += 8;
        }
    }

    #[inline]
    fn ensure_bits(&mut self, n: u8) -> bool {
        if self.bits_in < n {
            self.refill();
        }
        self.bits_in >= n
    }

    #[inline]
    pub fn peek_bits(&mut self, n: u8) -> Option<u16> {
        if n == 0 {
            return Some(0);
        }
        if !self.ensure_bits(n) {
            return None;
        }
        let shift = self.bits_in - n;
        Some(((self.bit_buf >> shift) & ((1u64 << n) - 1)) as u16)
    }

    #[inline]
    pub fn drop_bits(&mut self, n: u8) -> bool {
        if n == 0 {
            return true;
        }
        if !self.ensure_bits(n) {
            return false;
        }
        self.bits_in -= n;
        if self.bits_in == 0 {
            self.bit_buf = 0;
        } else {
            self.bit_buf &= (1u64 << self.bits_in) - 1;
        }
        true
    }

    #[inline]
    pub fn read_bits(&mut self, n: u8) -> Option<u16> {
        if n == 0 { return Some(0); }
        if n == 1 {
            return self.read_one_bit().map(|b| b as u16);
        }
        if !self.ensure_bits(n) {
            return None;
        }
        let shift = self.bits_in - n;
        let out = ((self.bit_buf >> shift) & ((1u64 << n) - 1)) as u16;
        self.bits_in -= n;
        if self.bits_in == 0 {
            self.bit_buf = 0;
        } else {
            self.bit_buf &= (1u64 << self.bits_in) - 1;
        }
        Some(out)
    }

    #[inline]
    pub fn read_one_bit(&mut self) -> Option<u8> {
        if self.bits_in == 0 {
            self.refill();
            if self.bits_in == 0 {
                return None;
            }
        }
        self.bits_in -= 1;
        let bit = ((self.bit_buf >> self.bits_in) & 1) as u8;
        if self.bits_in == 0 {
            self.bit_buf = 0;
        }
        Some(bit)
    }

    pub fn byte_position(&self) -> usize { self.pos }
}

// ---------------------------------------------------------------------------
// Huffman symbol decode via prebuilt binary tree
// ---------------------------------------------------------------------------

fn decode_sym(reader: &mut BitReader, tree: &DecodeTree) -> Option<u8> {
    if reader.ensure_bits(FAST_BITS) {
        let shift = reader.bits_in - FAST_BITS;
        let prefix = ((reader.bit_buf >> shift) & ((1u64 << FAST_BITS) - 1)) as usize;
        let (sym, len) = tree.fast[prefix as usize];
        if sym >= 0 {
            reader.bits_in -= len;
            if reader.bits_in == 0 {
                reader.bit_buf = 0;
            } else {
                reader.bit_buf &= (1u64 << reader.bits_in) - 1;
            }
            return Some(sym as u8);
        }
    }

    // Branch-reduced tree walk when we have enough buffered bits.
    if reader.ensure_bits(16) {
        let bits = reader.bit_buf;
        let mut idx = 0usize;
        for depth in 1u8..=16 {
            let bit_shift = reader.bits_in - depth;
            let bit = ((bits >> bit_shift) & 1) as usize;
            let next = tree.nodes[idx].child[bit];
            if next < 0 {
                return None;
            }
            idx = next as usize;
            let sym = tree.nodes[idx].sym;
            if sym >= 0 {
                reader.bits_in -= depth;
                if reader.bits_in == 0 {
                    reader.bit_buf = 0;
                } else {
                    reader.bit_buf &= (1u64 << reader.bits_in) - 1;
                }
                return Some(sym as u8);
            }
        }
        return None;
    }

    let mut idx = 0usize;
    for _ in 0..16 {
        let bit = reader.read_one_bit()? as usize;
        let next = tree.nodes[idx].child[bit];
        if next < 0 {
            return None;
        }
        idx = next as usize;
        let sym = tree.nodes[idx].sym;
        if sym >= 0 {
            return Some(sym as u8);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Adaptive arithmetic coding (current entropy coder)
// ---------------------------------------------------------------------------

/// Magic prefix of an arithmetic-coded plane payload.
const ARITH_MAGIC: [u8; 4] = *b"ARIT";
/// Version of the arithmetic payload layout.
const ARITH_CODEC_VERSION: u8 = 1;
/// Magic (4) + version (1) + config (1).
const ARITH_HEADER_SIZE: usize = 6;

/// Default model update period (fold pending counts after every coded symbol).
pub(crate) const DEFAULT_UPDATE_PERIOD: u8 = 1;
/// Largest representable update period (config byte keeps bit 7 reserved).
pub(crate) const MAX_UPDATE_PERIOD: u8 = 127;

/// Read the model update period from the environment.
///
/// `BITGRAIN_ARITH_UPDATE_PERIOD` selects the fold cadence: `0` keeps the models static
/// (uniform code, never updated), `1` updates after every symbol, `N` folds every N symbols.
/// Missing or malformed values fall back to [`DEFAULT_UPDATE_PERIOD`]; values above
/// [`MAX_UPDATE_PERIOD`] are clamped.
pub(crate) fn arithmetic_update_period() -> u8 {
    match std::env::var("BITGRAIN_ARITH_UPDATE_PERIOD") {
        Ok(v) => match v.trim().parse::<u32>() {
            Ok(n) => n.min(MAX_UPDATE_PERIOD as u32) as u8,
            Err(_) => DEFAULT_UPDATE_PERIOD,
        },
        Err(_) => DEFAULT_UPDATE_PERIOD,
    }
}

/// Upper bound of a frequency-table total before rescaling.
const MODEL_MAX_TOTAL: u32 = 32768;

/// Fixed-capacity adaptive frequency model.
///
/// `freq[..n]` holds the current frequencies (each ≥ 1); `pending[..n]` accumulates counts
/// between folds when `period > 1`. The update rule is shared verbatim by the encoder and the
/// decoder, which is what keeps both sides symmetric.
#[derive(Clone)]
struct Model {
    freq: [u16; 64],
    pending: [u16; 64],
    n: usize,
    total: u32,
    pending_count: u32,
    period: u8,
}

impl Model {
    fn new(n: usize, period: u8) -> Self {
        debug_assert!(n >= 2 && n <= 64);
        let mut freq = [0u16; 64];
        for f in freq[..n].iter_mut() {
            *f = 1;
        }
        Self {
            freq,
            pending: [0u16; 64],
            n,
            total: n as u32,
            pending_count: 0,
            period,
        }
    }

    /// Cumulative frequency below `sym`, its frequency, and the total.
    #[inline]
    fn cum_range(&self, sym: usize) -> (u64, u64, u64) {
        let mut cum = 0u64;
        for &f in &self.freq[..sym] {
            cum += f as u64;
        }
        let f = self.freq[sym] as u64;
        (cum, cum + f, self.total as u64)
    }

    /// Fold pending counts into the live table (identical on both sides).
    #[inline]
    fn fold(&mut self) {
        if self.pending_count == 0 {
            return;
        }
        for i in 0..self.n {
            let add = self.pending[i];
            if add != 0 {
                self.freq[i] = self.freq[i].saturating_add(add);
                self.pending[i] = 0;
            }
        }
        self.total += self.pending_count;
        self.pending_count = 0;

        // Keep totals bounded so `range * total` never overflows and every symbol keeps a
        // non-zero frequency.
        if self.total > MODEL_MAX_TOTAL {
            self.rescale();
        }
    }

    /// Halve all frequencies, keeping them ≥ 1 and the total bounded.
    #[inline]
    fn rescale(&mut self) {
        let mut total = 0u32;
        for i in 0..self.n {
            let f = ((self.freq[i] as u32 + 1) >> 1) as u16;
            self.freq[i] = f;
            total += f as u32;
        }
        self.total = total;
    }

    /// Update the model after coding `sym` (no-op for static models).
    #[inline]
    fn update(&mut self, sym: usize) {
        match self.period {
            0 => {}
            1 => {
                self.freq[sym] += 1;
                self.total += 1;
                if self.total > MODEL_MAX_TOTAL {
                    self.rescale();
                }
            }
            p => {
                self.pending[sym] += 1;
                self.pending_count += 1;
                if self.pending_count >= p as u32 {
                    self.fold();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bit-oriented arithmetic coder (32-bit code values, deferred underflow)
// ---------------------------------------------------------------------------

const HALF: u64 = 0x8000_0000;
const QUARTER: u64 = 0x4000_0000;
const THREE_QUARTER: u64 = 0xC000_0000;
const MASK32: u64 = 0xFFFF_FFFF;

/// MSB-first bit sink; pads the final partial byte with zero bits.
struct BitOut {
    buf: Vec<u8>,
    cur: u8,
    nbits: u8,
}

impl BitOut {
    fn new() -> Self {
        Self { buf: Vec::with_capacity(4096), cur: 0, nbits: 0 }
    }

    #[inline]
    fn put(&mut self, bit: bool) {
        self.cur = (self.cur << 1) | bit as u8;
        self.nbits += 1;
        if self.nbits == 8 {
            self.buf.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.buf.push(self.cur << (8 - self.nbits));
        }
        self.buf
    }
}

/// MSB-first bit source; bits past the end read as zero (arithmetic-coder padding convention).
struct BitIn<'a> {
    buf: &'a [u8],
    pos: usize,
    bit: u8,
}

impl<'a> BitIn<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0, bit: 0 }
    }

    #[inline]
    fn get(&mut self) -> u64 {
        if self.pos >= self.buf.len() {
            return 0;
        }
        let b = ((self.buf[self.pos] >> (7 - self.bit)) & 1) as u64;
        self.bit += 1;
        if self.bit == 8 {
            self.bit = 0;
            self.pos += 1;
        }
        b
    }
}

struct ArithEncoder {
    low: u64,
    high: u64,
    pending: u32,
    out: BitOut,
}

impl ArithEncoder {
    fn new() -> Self {
        Self { low: 0, high: MASK32, pending: 0, out: BitOut::new() }
    }

    #[inline]
    fn bit_follow(&mut self, bit: bool) {
        self.out.put(bit);
        let inv = !bit;
        while self.pending > 0 {
            self.out.put(inv);
            self.pending -= 1;
        }
    }

    #[inline]
    fn renorm(&mut self) {
        loop {
            if self.high < HALF {
                self.bit_follow(false);
            } else if self.low >= HALF {
                self.bit_follow(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low = (self.low << 1) & MASK32;
            self.high = ((self.high << 1) | 1) & MASK32;
        }
    }

    #[inline]
    fn encode_range(&mut self, cum_low: u64, cum_high: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high = self.low + range * cum_high / total - 1;
        self.low += range * cum_low / total;
        self.renorm();
    }

    #[inline]
    fn encode_bit(&mut self, model: &mut Model, bit: bool) {
        let sym = bit as usize;
        let (cl, ch, total) = model.cum_range(sym);
        self.encode_range(cl, ch, total);
        model.update(sym);
    }

    #[inline]
    fn encode_sym(&mut self, model: &mut Model, sym: usize) {
        let (cl, ch, total) = model.cum_range(sym);
        self.encode_range(cl, ch, total);
        model.update(sym);
    }

    /// Uniform binary split (exactly 1/2 each): the `nbits` low magnitude bits / raw payload.
    #[inline]
    fn encode_raw(&mut self, value: u32, nbits: u8) {
        let mut i = nbits;
        while i > 0 {
            i -= 1;
            let bit = ((value >> i) & 1) != 0;
            let range = self.high - self.low + 1;
            let half = range >> 1;
            if bit {
                self.low += half;
            } else {
                self.high = self.low + half - 1;
            }
            self.renorm();
        }
    }

    /// Emit the disambiguating bit plus all pending underflow bits, then pad the byte.
    fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < QUARTER {
            self.bit_follow(false);
        } else {
            self.bit_follow(true);
        }
        self.out.finish()
    }
}

struct ArithDecoder<'a> {
    low: u64,
    high: u64,
    code: u64,
    inp: BitIn<'a>,
}

impl<'a> ArithDecoder<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut inp = BitIn::new(buf);
        let mut code = 0u64;
        for _ in 0..32 {
            code = (code << 1) | inp.get();
        }
        Self { low: 0, high: MASK32, code, inp }
    }

    #[inline]
    fn renorm(&mut self) {
        loop {
            if self.high < HALF {
                // nothing
            } else if self.low >= HALF {
                self.code -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.code -= QUARTER;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low = (self.low << 1) & MASK32;
            self.high = ((self.high << 1) | 1) & MASK32;
            self.code = ((self.code << 1) & MASK32) | self.inp.get();
        }
    }

    #[inline]
    fn decode_sym(&mut self, model: &mut Model) -> usize {
        let range = self.high - self.low + 1;
        let total = model.total as u64;
        // Classic inverse of `encode_range`: maps `code` into [0, total-1].
        let scaled = (((self.code - self.low + 1) * total) - 1) / range;

        let mut cum = 0u64;
        let mut sym = model.n - 1;
        for i in 0..model.n {
            let f = model.freq[i] as u64;
            if scaled < cum + f {
                sym = i;
                break;
            }
            cum += f;
        }
        let f = model.freq[sym] as u64;
        self.high = self.low + range * (cum + f) / total - 1;
        self.low += range * cum / total;
        self.renorm();
        model.update(sym);
        sym
    }

    #[inline]
    fn decode_bit(&mut self, model: &mut Model) -> bool {
        self.decode_sym(model) != 0
    }

    #[inline]
    fn decode_raw(&mut self, nbits: u8) -> u32 {
        let mut value = 0u32;
        let mut i = nbits;
        while i > 0 {
            i -= 1;
            let range = self.high - self.low + 1;
            let half = range >> 1;
            let bit = self.code - self.low >= half;
            if bit {
                self.low += half;
                value |= 1 << i;
            } else {
                self.high = self.low + half - 1;
            }
            self.renorm();
        }
        value
    }
}

/// Encoder/decoder abstraction used by the shared block walker.
///
/// The encoder consumes the supplied `sym`/`bit`/`value`; the decoder ignores them and returns
/// what it decoded. `process_block` therefore has exactly one code path for both directions.
trait Coder {
    const ENCODE: bool;
    fn code_sym(&mut self, model: &mut Model, sym: usize) -> usize;
    fn code_bit(&mut self, model: &mut Model, bit: bool) -> bool;
    fn code_raw(&mut self, value: u32, nbits: u8) -> u32;
}

impl Coder for ArithEncoder {
    const ENCODE: bool = true;

    #[inline]
    fn code_sym(&mut self, model: &mut Model, sym: usize) -> usize {
        self.encode_sym(model, sym);
        sym
    }

    #[inline]
    fn code_bit(&mut self, model: &mut Model, bit: bool) -> bool {
        self.encode_bit(model, bit);
        bit
    }

    #[inline]
    fn code_raw(&mut self, value: u32, nbits: u8) -> u32 {
        self.encode_raw(value, nbits);
        value
    }
}

impl<'a> Coder for ArithDecoder<'a> {
    const ENCODE: bool = false;

    #[inline]
    fn code_sym(&mut self, model: &mut Model, _sym: usize) -> usize {
        self.decode_sym(model)
    }

    #[inline]
    fn code_bit(&mut self, model: &mut Model, _bit: bool) -> bool {
        self.decode_bit(model)
    }

    #[inline]
    fn code_raw(&mut self, _value: u32, nbits: u8) -> u32 {
        self.decode_raw(nbits)
    }
}

// ---------------------------------------------------------------------------
// Context models and the shared block walker
// ---------------------------------------------------------------------------

const NZ_CONTEXTS: usize = 32; // 8 zigzag bands × 4 zero-run buckets
const CAT_CONTEXTS: usize = 32; // 8 zigzag bands × 4 previous-category buckets
const LAST_NZ_CONTEXTS: usize = 4; // bucketed previous-block last_nz
const DC_CAT_CONTEXTS: usize = 4; // bucketed previous DC category

struct PlaneModels {
    last_nz: [Model; LAST_NZ_CONTEXTS],
    nz: [Model; NZ_CONTEXTS],
    cat: [Model; CAT_CONTEXTS],
    sign: Model,
    dc_cat: [Model; DC_CAT_CONTEXTS],
    dc_sign: Model,
}

impl PlaneModels {
    fn new(period: u8) -> Self {
        Self {
            last_nz: core::array::from_fn(|_| Model::new(64, period)),
            nz: core::array::from_fn(|_| Model::new(2, period)),
            cat: core::array::from_fn(|_| Model::new(16, period)),
            sign: Model::new(2, period),
            dc_cat: core::array::from_fn(|_| Model::new(17, period)),
            dc_sign: Model::new(2, period),
        }
    }
}

/// Per-plane adaptive context carried from block to block.
struct PlaneCtx {
    prev_dc: i16,
    prev_dc_cat: usize,
    prev_last_bucket: usize,
}

impl PlaneCtx {
    fn new() -> Self {
        Self { prev_dc: 0, prev_dc_cat: 0, prev_last_bucket: 0 }
    }
}

#[inline]
fn floor_log2(m: u32) -> u8 {
    debug_assert!(m >= 1);
    31 - m.leading_zeros() as u8
}

#[inline]
fn bucket_last_nz(last: usize) -> usize {
    match last {
        0 => 0,
        1..=3 => 1,
        4..=15 => 2,
        _ => 3,
    }
}

/// Walk one 8×8 block through the shared coder.
///
/// With `C::ENCODE == true` the coefficient values are read from `block`; otherwise the decoded
/// values are written into it. Every coding decision (models, contexts, update order) is shared.
fn process_block<C: Coder>(
    coder: &mut C,
    models: &mut PlaneModels,
    block: &mut Block,
    ctx: &mut PlaneCtx,
    use_dc_delta: bool,
) {
    // --- DC -------------------------------------------------------------
    let dc_known: i32 = if C::ENCODE { block.data[ZIGZAG[0]] as i32 } else { 0 };
    let dc_emit_known: i32 = if C::ENCODE {
        if use_dc_delta {
            block.data[ZIGZAG[0]].wrapping_sub(ctx.prev_dc) as i32
        } else {
            dc_known
        }
    } else {
        0
    };
    let dc_abs = dc_emit_known.unsigned_abs();
    let dc_cat_known = if dc_abs == 0 { 0 } else { (floor_log2(dc_abs) + 1) as usize };
    let dc_cat = coder.code_sym(&mut models.dc_cat[ctx.prev_dc_cat], dc_cat_known);

    // Magnitude is split as: implicit leading 1 (for cat > 0) + (cat-1) raw low bits.
    // The encoder derives the raw bits from its own category; the decoder uses the decoded one.
    let (dc_raw_known, dc_neg_known) = if C::ENCODE && dc_cat_known > 0 {
        let low = dc_abs - (1u32 << (dc_cat_known - 1));
        (low, dc_emit_known < 0)
    } else {
        (0, false)
    };
    let dc_raw = if dc_cat > 0 {
        debug_assert!(!C::ENCODE || dc_cat == dc_cat_known);
        coder.code_raw(dc_raw_known, (dc_cat - 1) as u8)
    } else {
        0
    };
    let dc_neg = if dc_cat > 0 {
        coder.code_bit(&mut models.dc_sign, dc_neg_known)
    } else {
        false
    };

    let dc_mag = if dc_cat == 0 { 0 } else { (1u32 << (dc_cat - 1)) | dc_raw };
    let dc_emit = if dc_neg { -(dc_mag as i32) } else { dc_mag as i32 };
    let dc_val: i16 = if use_dc_delta {
        ctx.prev_dc.wrapping_add(dc_emit as i16)
    } else {
        dc_emit as i16
    };
    if !C::ENCODE {
        block.data[ZIGZAG[0]] = dc_val;
    }
    ctx.prev_dc = dc_val;
    ctx.prev_dc_cat = dc_cat.min(3);

    // --- AC -------------------------------------------------------------
    let last_known: usize = if C::ENCODE {
        let mut last = 0usize;
        for zi in (1..64).rev() {
            if block.data[ZIGZAG[zi]] != 0 {
                last = zi;
                break;
            }
        }
        last
    } else {
        0
    };
    let last = coder.code_sym(&mut models.last_nz[ctx.prev_last_bucket], last_known);

    let mut run = 0usize;
    let mut prev_cat = 0usize;
    for zi in 1..=last {
        let band = (zi - 1) >> 3;
        let nz = if zi == last {
            true
        } else {
            let known = if C::ENCODE { block.data[ZIGZAG[zi]] != 0 } else { false };
            coder.code_bit(&mut models.nz[band * 4 + run.min(3)], known)
        };
        if !nz {
            run += 1;
            continue;
        }

        let v_known: i32 = if C::ENCODE { block.data[ZIGZAG[zi]] as i32 } else { 0 };
        let abs_known = v_known.unsigned_abs();
        let cat_known = if C::ENCODE { floor_log2(abs_known) as usize } else { 0 };
        let cat = coder.code_sym(&mut models.cat[band * 4 + prev_cat.min(3)], cat_known);

        let (raw_known, neg_known) = if C::ENCODE {
            (abs_known - (1u32 << cat_known), v_known < 0)
        } else {
            (0, false)
        };
        debug_assert!(!C::ENCODE || cat == cat_known);
        let raw = coder.code_raw(raw_known, cat as u8);
        let neg = coder.code_bit(&mut models.sign, neg_known);

        let mag = (1u32 << cat) | raw;
        let v = if neg { -(mag as i32) } else { mag as i32 };
        if !C::ENCODE {
            block.data[ZIGZAG[zi]] = v as i16;
        }
        prev_cat = cat;
        run = 0;
    }
    ctx.prev_last_bucket = bucket_last_nz(last);
}

fn arith_encode_plane(
    blocks: &[Block],
    use_dc_delta: bool,
    period: u8,
) -> Vec<u8> {
    let mut enc = ArithEncoder::new();
    let mut models = PlaneModels::new(period);
    let mut ctx = PlaneCtx::new();
    for block in blocks {
        let mut copy = *block;
        process_block(&mut enc, &mut models, &mut copy, &mut ctx, use_dc_delta);
    }
    let payload = enc.finish();

    let mut data = Vec::with_capacity(ARITH_HEADER_SIZE + payload.len());
    data.extend_from_slice(&ARITH_MAGIC);
    data.push(ARITH_CODEC_VERSION);
    data.push(period & 0x7F);
    data.extend_from_slice(&payload);

    let mut out = Vec::with_capacity(4 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

fn arith_decode_plane(
    payload: &[u8],
    n_blocks: usize,
    use_dc_delta: bool,
    period: u8,
) -> Option<Vec<Block>> {
    let body = payload.get(ARITH_HEADER_SIZE..)?;
    let mut dec = ArithDecoder::new(body);
    let mut models = PlaneModels::new(period);
    let mut ctx = PlaneCtx::new();
    let mut blocks = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let mut block = Block::new();
        process_block(&mut dec, &mut models, &mut block, &mut ctx, use_dc_delta);
        blocks.push(block);
    }
    Some(blocks)
}

// ---------------------------------------------------------------------------
// Encode / decode a full plane of blocks
// ---------------------------------------------------------------------------

/// Encode all blocks of a plane into a Vec<u8> prefixed with its byte length (u32 LE).
/// Format: [len: u32 LE][payload]
/// The payload is arithmetic-coded (see module docs); the length prefix lets the decoder skip
/// exactly to the next plane boundary.
pub fn encode_plane(blocks: &[Block], is_chroma: bool) -> Vec<u8> {
    encode_plane_with_profile(blocks, is_chroma, false, false)
}

pub fn encode_plane_with_ac(blocks: &[Block], is_chroma: bool, use_chroma_ac: bool) -> Vec<u8> {
    encode_plane_with_profile(blocks, is_chroma, use_chroma_ac, false)
}

pub fn encode_plane_with_profile(
    blocks: &[Block],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Vec<u8> {
    encode_plane_with_period(blocks, is_chroma, use_chroma_ac, use_dc_delta, arithmetic_update_period())
}

/// Same as [`encode_plane_with_profile`] with an explicit model update period.
/// `0` = static models, `1` = update after every symbol, `N` = fold every N symbols.
pub(crate) fn encode_plane_with_period(
    blocks: &[Block],
    _is_chroma: bool,
    _use_chroma_ac: bool,
    use_dc_delta: bool,
    period: u8,
) -> Vec<u8> {
    arith_encode_plane(blocks, use_dc_delta, period.min(MAX_UPDATE_PERIOD))
}

/// Legacy Huffman plane encoder (pre-2.1 streams). Kept for reference/tests only; the public
/// encoder above now emits arithmetic-coded payloads.
#[cfg(test)]
pub(crate) fn legacy_encode_plane_with_profile(
    blocks: &[Block],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Vec<u8> {
    let dc_table = if is_chroma { CHROMA_DC_TABLE } else { LUMA_DC_TABLE };
    let ac_table = if use_chroma_ac { jpeg_chroma_ac_table() } else { jpeg_ac_table() };
    let eob = ac_table[0x00];
    let zrl = ac_table[0xF0];
    let mut w = BitWriter::new();
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
        let (dc_len, dc_code) = dc_table[dc_cat as usize];
        w.write_bits(dc_code, dc_len);
        if dc_cat > 0 { w.write_bits(magnitude_bits(dc_emit, dc_cat), dc_cat); }

        // AC (JPEG-style optimization): stop at last non-zero and emit EOB only when needed.
        let mut last_nz = 63usize;
        while last_nz > 0 && block.data[ZIGZAG[last_nz]] == 0 {
            last_nz -= 1;
        }
        if last_nz == 0 {
            w.write_bits(eob.1, eob.0);
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
                w.write_bits(zrl.1, zrl.0);
                zero_run -= 16;
            }
            let cat = category(val);
            let sym = (zero_run << 4) | cat;
            let (al, ac) = ac_table[sym as usize];
            debug_assert!(al > 0, "missing AC Huffman for RS {sym:#04x} (run={zero_run} cat={cat})");
            if al == 0 {
                // Defensive fallback for unexpected table/profile mismatch.
                let (el, ec) = ac_table[0x00];
                w.write_bits(ec, el);
                break;
            }
            w.write_bits(ac, al);
            w.write_bits(magnitude_bits(val, cat), cat);
            zero_run = 0;
        }
        w.write_bits(eob.1, eob.0);
    }
    w.flush();

    // Prepend 4-byte length so decoder can skip exactly to next plane
    let data = w.buf;
    let len = data.len() as u32;
    let mut out = Vec::with_capacity(4 + data.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&data);
    out
}

/// Decode a plane of `n_blocks` blocks from `buf[start..]`.
/// Reads the 4-byte length prefix, then decodes exactly that many bytes.
/// Returns (blocks, new_byte_position) or None on error.
pub fn decode_plane(buf: &[u8], start: usize, n_blocks: usize, is_chroma: bool) -> Option<(Vec<Block>, usize)> {
    decode_plane_with_profile(buf, start, n_blocks, is_chroma, false, false)
}

pub fn decode_plane_with_ac(
    buf: &[u8],
    start: usize,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
) -> Option<(Vec<Block>, usize)> {
    decode_plane_with_profile(buf, start, n_blocks, is_chroma, use_chroma_ac, false)
}

pub fn decode_plane_with_profile(
    buf: &[u8],
    start: usize,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Option<(Vec<Block>, usize)> {
    if start + 4 > buf.len() { return None; }
    let plane_len = u32::from_le_bytes(buf[start..start+4].try_into().unwrap()) as usize;
    let data_start = start + 4;
    let data_end   = data_start.checked_add(plane_len)?;
    if data_end > buf.len() {
        return None;
    }
    let data = &buf[data_start..data_end];

    // Arithmetic payload: [magic][version][config][range-coded body].
    if data.len() >= 4 && data[..4] == ARITH_MAGIC {
        if data.len() < ARITH_HEADER_SIZE || data[4] != ARITH_CODEC_VERSION {
            return None;
        }
        let period = data[5] & 0x7F;
        let blocks = arith_decode_plane(data, n_blocks, use_dc_delta, period)?;
        return Some((blocks, data_end));
    }

    // Legacy Huffman payload (written by bitgrain <= 2.0.0).
    let dc_tree = if is_chroma { chroma_dc_tree() } else { luma_dc_tree() };
    let ac_tree = ac_tree(use_chroma_ac);

    let mut reader = BitReader::new(data, 0);
    let blocks = decode_plane_blocks(&mut reader, n_blocks, dc_tree, ac_tree, use_dc_delta)?;

    // Return data_end as the next byte position (exact plane boundary)
    Some((blocks, data_end))
}

fn decode_plane_blocks(
    reader: &mut BitReader,
    n_blocks: usize,
    dc_tree: &DecodeTree,
    ac_tree: &DecodeTree,
    use_dc_delta: bool,
) -> Option<Vec<Block>> {
    let mut blocks = Vec::with_capacity(n_blocks);
    let mut prev_dc: i16 = 0;
    for _bi in 0..n_blocks {
        let mut block = Block::new();

        // DC
        let dc_cat = match decode_sym(reader, dc_tree) {
            Some(c) => c,
            None => return None,
        };
        let dc_diff = if dc_cat == 0 {
            0i16
        } else {
            let bits = match reader.read_bits(dc_cat) {
                Some(b) => b,
                None => return None,
            };
            magnitude_decode(bits, dc_cat)
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
            let sym = match decode_sym(reader, ac_tree) {
                Some(s) => s,
                None => return None,
            };
            if sym == 0x00 { break; } // EOB
            if sym == 0xF0 {
                // ZRL: exactly 16 consecutive zeros.
                if ac_idx + 16 > 64 { return None; }
                ac_idx += 16;
                continue;
            }
            let run = (sym >> 4) as usize;
            let cat = sym & 0x0F;
            // Need room for run zeros plus one non-zero coefficient.
            if ac_idx + run >= 64 { return None; }
            ac_idx += run;
            let bits = match reader.read_bits(cat) {
                Some(b) => b,
                None => return None,
            };
            block.data[ZIGZAG[ac_idx]] = magnitude_decode(bits, cat);
            ac_idx += 1;
            if ac_idx > 64 { return None; }
        }

        blocks.push(block);
    }

    Some(blocks)
}

