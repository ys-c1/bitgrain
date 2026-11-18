//! Zero-order adaptive arithmetic coding backend (entropy backend v1).
//!
//! One independent order-0 model per [`SymClass`]; frequencies adapt as
//! symbols stream through (Fenwick tree, periodic halving). Raw magnitude
//! bits go through a fixed uniform binary model, which costs exactly one
//! bit per bit — the same bypass semantics as the Huffman backend.
//!
//! The coder is the classic Witten–Neal–Cleary arithmetic coder with
//! 32-bit code values and E1/E2/E3 renormalization. Bits are packed
//! MSB-first with no 0xFF stuffing (planes are length-prefixed, so no
//! marker emulation is needed). The wire format is self-terminating:
//! the decoder knows the exact symbol count from the block count.
//!
//! A future v2 (context-adaptive, higher-order, rANS, ...) implements the
//! same `EntropyEncoder`/`EntropyDecoder` traits; the encoder main flow
//! does not change.

use crate::block::Block;
use crate::entropy_coder::{decode_plane_walk, EntropyDecoder, EntropyEncoder, SymClass};

// ---------------------------------------------------------------------------
// Adaptive order-0 model (Fenwick tree over symbol frequencies)
// ---------------------------------------------------------------------------

const MAX_ALPHABET: usize = 256;
/// Frequency increment per observed symbol.
const INC: u32 = 16;
/// Rescale when the total would exceed this (keeps `tot` well under 2^15 so
/// 32-bit range arithmetic always has enough precision).
const MAX_TOT: u32 = 1 << 14;

struct Model {
    n: usize,
    freq: [u32; MAX_ALPHABET],
    tree: [u32; MAX_ALPHABET + 1], // 1-based Fenwick tree
    tot: u32,
}

impl Model {
    fn new(n: usize) -> Self {
        let mut m = Model { n, freq: [0; MAX_ALPHABET], tree: [0; MAX_ALPHABET + 1], tot: 0 };
        for f in m.freq.iter_mut().take(n) {
            *f = 1;
        }
        m.rebuild();
        m
    }

    fn rebuild(&mut self) {
        self.tree = [0; MAX_ALPHABET + 1];
        self.tot = 0;
        for i in 0..self.n {
            self.tot += self.freq[i];
            let mut j = i + 1;
            while j <= self.n {
                self.tree[j] += self.freq[i];
                j += j & j.wrapping_neg();
            }
        }
    }

    /// Cumulative frequency of all symbols before `sym`.
    #[inline]
    fn cum_before(&self, sym: usize) -> u32 {
        let mut s = 0u32;
        let mut i = sym; // prefix over tree[1..=sym]
        while i > 0 {
            s += self.tree[i];
            i &= i - 1;
        }
        s
    }

    /// Find the symbol whose cumulative interval contains `v` (v < tot).
    /// Returns (sym, cum_before, freq).
    fn find(&self, v: u32) -> (usize, u32, u32) {
        let mut idx = 0usize; // 1-based Fenwick position
        let mut cum = 0u32;
        let mut bit = {
            let mut b = 1usize;
            while b << 1 <= self.n {
                b <<= 1;
            }
            b
        };
        while bit != 0 {
            let next = idx + bit;
            if next <= self.n && cum + self.tree[next] <= v {
                idx = next;
                cum += self.tree[next];
            }
            bit >>= 1;
        }
        // idx symbols sit entirely below v; the next one contains it.
        let sym = idx; // 0-based
        (sym, cum, self.freq[sym])
    }

    #[inline]
    fn update(&mut self, sym: usize) {
        if self.tot + INC > MAX_TOT {
            for f in self.freq.iter_mut().take(self.n) {
                *f = (*f + 1) >> 1;
            }
            self.rebuild();
        }
        self.freq[sym] += INC;
        self.tot += INC;
        let mut j = sym + 1;
        while j <= self.n {
            self.tree[j] += INC;
            j += j & j.wrapping_neg();
        }
    }
}

// ---------------------------------------------------------------------------
// Bit sink / source (MSB-first, no byte stuffing)
// ---------------------------------------------------------------------------

struct BitSink {
    buf: Vec<u8>,
    acc: u8,
    nbits: u8,
}

impl BitSink {
    #[inline]
    fn push_bit(&mut self, b: u32) {
        self.acc = (self.acc << 1) | (b as u8 & 1);
        self.nbits += 1;
        if self.nbits == 8 {
            self.buf.push(self.acc);
            self.acc = 0;
            self.nbits = 0;
        }
    }

    fn flush_pad_zeros(&mut self) {
        if self.nbits > 0 {
            self.buf.push(self.acc << (8 - self.nbits));
            self.acc = 0;
            self.nbits = 0;
        }
    }
}

struct BitSource<'a> {
    buf: &'a [u8],
    pos: usize,
    acc: u8,
    nbits: u8,
}

impl<'a> BitSource<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0, acc: 0, nbits: 0 }
    }

    /// Past the end of the stream, yield 0 bits (standard arithmetic-decoder
    /// behavior; the encoder emitted enough bits for every real symbol).
    #[inline]
    fn next_bit(&mut self) -> u32 {
        if self.nbits == 0 {
            self.acc = if self.pos < self.buf.len() {
                let b = self.buf[self.pos];
                self.pos += 1;
                b
            } else {
                0
            };
            self.nbits = 8;
        }
        self.nbits -= 1;
        ((self.acc >> self.nbits) & 1) as u32
    }
}

// ---------------------------------------------------------------------------
// Witten–Neal–Cleary arithmetic coder cores (32-bit, E1/E2/E3 scaling)
// ---------------------------------------------------------------------------

const HALF: u32 = 0x8000_0000;
const FIRST_QTR: u32 = 0x4000_0000;
const THIRD_QTR: u32 = 0xC000_0000;

struct EncCore {
    low: u32,
    high: u32,
    pending: u64,
    sink: BitSink,
}

impl EncCore {
    fn new() -> Self {
        Self {
            low: 0,
            high: u32::MAX,
            pending: 0,
            sink: BitSink { buf: Vec::with_capacity(4096), acc: 0, nbits: 0 },
        }
    }

    fn reset(&mut self) {
        self.low = 0;
        self.high = u32::MAX;
        self.pending = 0;
        self.sink.buf.clear();
        self.sink.acc = 0;
        self.sink.nbits = 0;
        // Reserve the 4-byte plane-length prefix in place; patched by finish().
        self.sink.buf.extend_from_slice(&[0, 0, 0, 0]);
    }

    #[inline]
    fn emit(&mut self, bit: u32) {
        self.sink.push_bit(bit);
        while self.pending > 0 {
            self.sink.push_bit(bit ^ 1);
            self.pending -= 1;
        }
    }

    #[inline]
    fn encode_interval(&mut self, cum: u32, freq: u32, tot: u32) {
        let range = (self.high - self.low) as u64 + 1;
        self.high = self.low + ((range * (cum + freq) as u64) / tot as u64 - 1) as u32;
        self.low = self.low + ((range * cum as u64) / tot as u64) as u32;
        loop {
            if self.high < HALF {
                self.emit(0);
            } else if self.low >= HALF {
                self.emit(1);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.pending += 1;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < FIRST_QTR {
            self.emit(0);
        } else {
            self.emit(1);
        }
        self.sink.flush_pad_zeros();
        let mut buf = self.sink.buf;
        let len = (buf.len() - 4) as u32;
        buf[0..4].copy_from_slice(&len.to_le_bytes());
        buf
    }
}

struct DecCore<'a> {
    low: u32,
    high: u32,
    value: u32,
    src: BitSource<'a>,
}

impl<'a> DecCore<'a> {
    fn new(buf: &'a [u8]) -> Self {
        let mut d = Self { low: 0, high: u32::MAX, value: 0, src: BitSource::new(buf) };
        for _ in 0..32 {
            d.value = (d.value << 1) | d.src.next_bit();
        }
        d
    }

    /// Scale the current code value into a cumulative-frequency target.
    #[inline]
    fn target(&self, tot: u32) -> u32 {
        let range = (self.high - self.low) as u64 + 1;
        ((((self.value - self.low) as u64 + 1) * tot as u64 - 1) / range) as u32
    }

    #[inline]
    fn decode_interval(&mut self, cum: u32, freq: u32, tot: u32) {
        let range = (self.high - self.low) as u64 + 1;
        self.high = self.low + ((range * (cum + freq) as u64) / tot as u64 - 1) as u32;
        self.low = self.low + ((range * cum as u64) / tot as u64) as u32;
        loop {
            if self.high < HALF {
                // E1: nothing to subtract
            } else if self.low >= HALF {
                self.value -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.value -= FIRST_QTR;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.value = (self.value << 1) | self.src.next_bit();
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder backend
// ---------------------------------------------------------------------------

/// Zero-order adaptive arithmetic encoder. One `init`/`finish` cycle encodes
/// one plane; models reset per plane, matching the Huffman backend.
pub struct ArithEncoder {
    core: EncCore,
    dc_luma: Model,
    dc_chroma: Model,
    ac_luma: Model,
    ac_chroma: Model,
}

impl ArithEncoder {
    pub fn new() -> Self {
        Self {
            core: EncCore::new(),
            dc_luma: Model::new(12),
            dc_chroma: Model::new(12),
            ac_luma: Model::new(256),
            ac_chroma: Model::new(256),
        }
    }
}

impl EntropyEncoder for ArithEncoder {
    fn init(&mut self) {
        self.core.reset();
        self.dc_luma = Model::new(12);
        self.dc_chroma = Model::new(12);
        self.ac_luma = Model::new(256);
        self.ac_chroma = Model::new(256);
    }

    #[inline]
    fn encode_symbol(&mut self, class: SymClass, sym: u8) {
        let m = match class {
            SymClass::DcLuma => &mut self.dc_luma,
            SymClass::DcChroma => &mut self.dc_chroma,
            SymClass::AcLuma => &mut self.ac_luma,
            SymClass::AcChroma => &mut self.ac_chroma,
        };
        let cum = m.cum_before(sym as usize);
        let freq = m.freq[sym as usize];
        let tot = m.tot;
        self.core.encode_interval(cum, freq, tot);
        m.update(sym as usize);
    }

    #[inline]
    fn encode_raw_bits(&mut self, bits: u16, n: u8) {
        // Uniform binary bypass: exactly one bit of code space per raw bit.
        for i in (0..n).rev() {
            let bit = ((bits >> i) & 1) as u32;
            self.core.encode_interval(bit, 1, 2);
        }
    }

    fn finish(&mut self) -> Vec<u8> {
        let core = std::mem::replace(&mut self.core, EncCore::new());
        core.finish()
    }
}

// ---------------------------------------------------------------------------
// Decoder backend
// ---------------------------------------------------------------------------

/// Zero-order adaptive arithmetic decoder, mirror of [`ArithEncoder`].
pub struct ArithDecoder<'a> {
    core: DecCore<'a>,
    dc_luma: Model,
    dc_chroma: Model,
    ac_luma: Model,
    ac_chroma: Model,
}

impl<'a> ArithDecoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            core: DecCore::new(buf),
            dc_luma: Model::new(12),
            dc_chroma: Model::new(12),
            ac_luma: Model::new(256),
            ac_chroma: Model::new(256),
        }
    }
}

impl<'a> EntropyDecoder for ArithDecoder<'a> {
    #[inline]
    fn decode_symbol(&mut self, class: SymClass) -> Option<u8> {
        let m = match class {
            SymClass::DcLuma => &mut self.dc_luma,
            SymClass::DcChroma => &mut self.dc_chroma,
            SymClass::AcLuma => &mut self.ac_luma,
            SymClass::AcChroma => &mut self.ac_chroma,
        };
        let v = self.core.target(m.tot);
        let (sym, cum, freq) = m.find(v);
        let tot = m.tot;
        self.core.decode_interval(cum, freq, tot);
        m.update(sym);
        Some(sym as u8)
    }

    #[inline]
    fn decode_raw_bits(&mut self, n: u8) -> Option<u16> {
        let mut out: u16 = 0;
        for _ in 0..n {
            // Uniform binary bypass, mirroring encode_raw_bits.
            let bit = self.core.target(2).min(1);
            self.core.decode_interval(bit, 1, 2);
            out = (out << 1) | bit as u16;
        }
        Some(out)
    }
}

/// Decode one arithmetic-coded plane from `buf[start..]`.
/// Reads the 4-byte length prefix, then decodes exactly `n_blocks` blocks.
/// Returns (blocks, new_byte_position) or None on error.
pub fn decode_plane_arith(
    buf: &[u8],
    start: usize,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Option<(Vec<Block>, usize)> {
    if start + 4 > buf.len() {
        return None;
    }
    let plane_len = u32::from_le_bytes(buf[start..start + 4].try_into().unwrap()) as usize;
    let data_start = start + 4;
    let data_end = data_start.checked_add(plane_len)?;
    if data_end > buf.len() {
        return None;
    }

    let mut dec = ArithDecoder::new(&buf[data_start..data_end]);
    let blocks = decode_plane_walk(&mut dec, n_blocks, is_chroma, use_chroma_ac, use_dc_delta)?;
    Some((blocks, data_end))
}
