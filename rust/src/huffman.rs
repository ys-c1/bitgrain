//! Static Huffman coding for DCT coefficients.
//!
//! Uses the standard JPEG Huffman tables (ISO 10918-1 Annex K):
//!   - Luma DC + full luminance AC for Y (and A); chroma DC for Cb/Cr; **same luminance AC** for
//!     all AC planes so .bg v4/v5 matches historical bitgrain streams.
//!
//! The bitstream is packed MSB-first, continuous across all blocks in a plane.
//! 0xFF bytes are stuffed as 0xFF 0x00 (JPEG convention).
//! A single flush (pad with 1s) is written at the end of each plane.

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
fn category(v: i16) -> u8 {
    if v == 0 { return 0; }
    (16 - v.unsigned_abs().leading_zeros()) as u8
}

#[inline]
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

    /// Resume reading at an arbitrary entropy-bit position recorded by the encoder:
    /// `byte_pos` is the byte containing the next unread bit and `bit_off` is the
    /// number of bits already consumed from it (MSB-first, 0..=7).
    ///
    /// The first byte is loaded manually, so JPEG byte-unstuffing is mirrored here:
    /// if it is 0xFF, the stuffed 0x00 that follows it is skipped.
    pub fn new_at_bit(buf: &'a [u8], byte_pos: usize, bit_off: u8) -> Option<Self> {
        if bit_off > 7 || byte_pos >= buf.len() {
            return None;
        }
        let byte = buf[byte_pos];
        let mut pos = byte_pos + 1;
        if byte == 0xFF && pos < buf.len() && buf[pos] == 0x00 {
            pos += 1;
        }
        let keep = 8 - bit_off;
        Some(Self {
            buf,
            pos,
            bit_buf: (byte as u64) & ((1u64 << keep) - 1),
            bits_in: keep,
        })
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
// Encode / decode a full plane of blocks
// ---------------------------------------------------------------------------

/// Encode all blocks of a plane into a Vec<u8> prefixed with its byte length (u32 LE).
/// Format: [len: u32 LE][bitstream bytes]
/// The length prefix allows the decoder to skip exactly to the next plane boundary.
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
    encode_plane_with_profile_indexed(blocks, is_chroma, use_chroma_ac, use_dc_delta).0
}

/// Encode all blocks of a plane (same bitstream as [`encode_plane_with_profile`])
/// and, while encoding, record a random-access restart point for **every** block:
/// the byte offset of the block's first entropy bit (relative to the start of the
/// returned plane buffer, i.e. including the 4-byte length prefix), the bit offset
/// of that bit inside the byte (MSB-first), and the DC predictor state with which
/// the decoder must start so the block decodes independently of its predecessors.
pub fn encode_plane_with_profile_indexed(
    blocks: &[Block],
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> (Vec<u8>, Vec<BlockRestart>) {
    let dc_table = if is_chroma { CHROMA_DC_TABLE } else { LUMA_DC_TABLE };
    let ac_table = if use_chroma_ac { jpeg_chroma_ac_table() } else { jpeg_ac_table() };
    let eob = ac_table[0x00];
    let zrl = ac_table[0xF0];
    let mut w = BitWriter::new();
    let mut prev_dc: i16 = 0;
    let mut restarts = Vec::with_capacity(blocks.len());

    for block in blocks {
        // Restart point for this block: its first bit lands in the next byte to be
        // emitted (w.buf.len()), after `w.bits_in` pending bits of the previous block.
        restarts.push(BlockRestart {
            byte_off: (4 + w.buf.len()) as u32,
            dc_pred: prev_dc,
            bit_off: w.bits_in,
        });

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
    (out, restarts)
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
    let data_end   = data_start + plane_len;
    if data_end > buf.len() {
        return None;
    }

    let dc_tree = if is_chroma { chroma_dc_tree() } else { luma_dc_tree() };
    let ac_tree = ac_tree(use_chroma_ac);
    let data = &buf[data_start..data_end];

    let mut reader = BitReader::new(data, 0);
    let blocks = decode_plane_blocks(&mut reader, n_blocks, dc_tree, ac_tree, use_dc_delta, 0)?;

    // Return data_end as the next byte position (exact plane boundary)
    Some((blocks, data_end))
}

/// Decode `n_blocks` consecutive blocks starting from a recorded restart point.
///
/// `stream` is the whole .bg file and `restart` carries absolute file offsets
/// (as stored in a parsed [`BgIndex`]). The blocks decode independently of every
/// preceding block: the bit reader resumes at the recorded entropy-bit position
/// and the DC predictor is seeded from `restart.dc_pred`.
pub fn decode_blocks_at(
    stream: &[u8],
    restart: &BlockRestart,
    n_blocks: usize,
    is_chroma: bool,
    use_chroma_ac: bool,
    use_dc_delta: bool,
) -> Option<Vec<Block>> {
    let mut reader = BitReader::new_at_bit(stream, restart.byte_off as usize, restart.bit_off)?;
    let dc_tree = if is_chroma { chroma_dc_tree() } else { luma_dc_tree() };
    let ac_tree = ac_tree(use_chroma_ac);
    decode_plane_blocks(&mut reader, n_blocks, dc_tree, ac_tree, use_dc_delta, restart.dc_pred)
}

fn decode_plane_blocks(
    reader: &mut BitReader,
    n_blocks: usize,
    dc_tree: &DecodeTree,
    ac_tree: &DecodeTree,
    use_dc_delta: bool,
    init_prev_dc: i16,
) -> Option<Vec<Block>> {
    let mut blocks = Vec::with_capacity(n_blocks);
    let mut prev_dc: i16 = init_prev_dc;
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


// ---------------------------------------------------------------------------
// Random-access block index (.bg sidecar)
// ---------------------------------------------------------------------------
//
// Motivation: the Huffman bitstream of a plane is continuous across blocks and
// (v10+) DC coefficients are delta-coded, so decoding block N previously
// required decoding blocks 0..N first. The encoder now records, for every
// block, the exact entropy-bit position and the DC predictor state needed to
// resume decoding there. A serialized `BgIndex` keeps one restart point per
// `stride` blocks (a "chunk"); a chunk decodes fully independently, which
// enables frame/region-level random preview without a from-the-start decode.
//
// Serialized layout (all integers little-endian):
//   "BGIX"            4 bytes  magic
//   version           1 byte   (= 1)
//   n_planes          1 byte
//   stride            u32      blocks per chunk (same for all planes)
//   per plane:
//     n_blocks        u32
//     n_entries       u32      (= ceil(n_blocks / stride))
//   entries           n_entries x 7 bytes, plane 0 first, in block order:
//     byte_off        u32      absolute offset in the .bg file of the byte
//                              holding the chunk's first entropy bit
//     dc_pred         i16      DC predictor state at chunk start
//     bit_off         u8       consumed bits (MSB-first) in that byte, 0..=7
//   crc32             u32      CRC-32 (IEEE) of every preceding byte
//
// Any corruption (bad magic/version, truncated or inconsistent structure,
// CRC mismatch) makes `BgIndex::parse` return `None`.

/// Restart point allowing independent decode of one block (and its successors).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockRestart {
    /// Byte offset of the block's first entropy bit. Relative to the plane
    /// buffer start (length prefix included) when produced by
    /// [`encode_plane_with_profile_indexed`]; absolute file offset once stored
    /// in a [`BgIndex`].
    pub byte_off: u32,
    /// DC predictor (previous block's quantized DC, 0 at plane start).
    pub dc_pred: i16,
    /// Bits already consumed from that byte, MSB-first (0..=7).
    pub bit_off: u8,
}

/// Random-access index for one plane: one restart point per `stride` blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneIndex {
    pub n_blocks: u32,
    /// Restart points for blocks 0, stride, 2*stride, ... (absolute offsets).
    pub entries: Vec<BlockRestart>,
}

/// Whole-stream random-access index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BgIndex {
    /// Blocks per chunk; chunk c of a plane covers blocks [c*stride, ...).
    pub stride: u32,
    pub planes: Vec<PlaneIndex>,
}

const BG_INDEX_MAGIC: &[u8; 4] = b"BGIX";
const BG_INDEX_VERSION: u8 = 1;

/// CRC-32 (IEEE 802.3, polynomial 0xEDB88320), computed over `data`.
/// Bitwise implementation: index blobs are small, so a table is not worth it.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

impl BgIndex {
    /// Build an index from per-plane restart points recorded at encode time.
    ///
    /// `planes` yields, per plane, `(plane_file_pos, restarts)` where
    /// `plane_file_pos` is the plane's byte offset in the .bg file and
    /// `restarts` are the plane-relative per-block restart points returned by
    /// [`encode_plane_with_profile_indexed`]. The smallest `stride` whose
    /// serialized index stays within `max_index_bytes` is chosen, so the index
    /// adapts to the stream's entropy (high-entropy streams get finer chunks).
    pub fn build(planes: &[(usize, Vec<BlockRestart>)], max_index_bytes: usize) -> BgIndex {
        let n_planes = planes.len();
        let total_blocks: usize = planes.iter().map(|(_, r)| r.len()).sum();
        let mut stride = 1usize;
        while stride <= total_blocks {
            let entries: usize = planes
                .iter()
                .map(|(_, r)| r.len().div_ceil(stride))
                .sum();
            let size = 4 + 1 + 1 + 4 + 8 * n_planes + 7 * entries + 4;
            if size <= max_index_bytes || stride >= total_blocks.max(1) {
                break;
            }
            stride += 1;
        }

        let planes = planes
            .iter()
            .map(|(plane_pos, restarts)| {
                let entries = restarts
                    .iter()
                    .step_by(stride)
                    .map(|r| BlockRestart {
                        byte_off: r.byte_off + *plane_pos as u32,
                        dc_pred: r.dc_pred,
                        bit_off: r.bit_off,
                    })
                    .collect();
                PlaneIndex {
                    n_blocks: restarts.len() as u32,
                    entries,
                }
            })
            .collect();
        BgIndex {
            stride: stride as u32,
            planes,
        }
    }

    /// Locate the chunk containing block `block_idx` of plane `plane`.
    /// Returns `(restart, first_block, n_blocks_in_chunk)`.
    pub fn locate(&self, plane: usize, block_idx: usize) -> Option<(BlockRestart, usize, usize)> {
        let p = self.planes.get(plane)?;
        let stride = self.stride as usize;
        if stride == 0 || block_idx >= p.n_blocks as usize {
            return None;
        }
        let chunk = block_idx / stride;
        let restart = *p.entries.get(chunk)?;
        let first = chunk * stride;
        let len = stride.min(p.n_blocks as usize - first);
        Some((restart, first, len))
    }

    /// Serialized size in bytes (== `self.serialize().len()` without allocating).
    pub fn serialized_size(&self) -> usize {
        let entries: usize = self.planes.iter().map(|p| p.entries.len()).sum();
        4 + 1 + 1 + 4 + 8 * self.planes.len() + 7 * entries + 4
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.serialized_size());
        out.extend_from_slice(BG_INDEX_MAGIC);
        out.push(BG_INDEX_VERSION);
        out.push(self.planes.len() as u8);
        out.extend_from_slice(&self.stride.to_le_bytes());
        for p in &self.planes {
            out.extend_from_slice(&p.n_blocks.to_le_bytes());
            out.extend_from_slice(&(p.entries.len() as u32).to_le_bytes());
        }
        for p in &self.planes {
            for e in &p.entries {
                out.extend_from_slice(&e.byte_off.to_le_bytes());
                out.extend_from_slice(&e.dc_pred.to_le_bytes());
                out.push(e.bit_off);
            }
        }
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parse and validate an index blob. Returns `None` on any corruption:
    /// bad magic/version, truncation, structural inconsistency, out-of-range
    /// or non-monotonic offsets, or CRC mismatch.
    pub fn parse(data: &[u8]) -> Option<BgIndex> {
        if data.len() < 4 + 1 + 1 + 4 + 4 {
            return None;
        }
        if &data[0..4] != BG_INDEX_MAGIC || data[4] != BG_INDEX_VERSION {
            return None;
        }
        let n_planes = data[5] as usize;
        if n_planes == 0 || n_planes > 8 {
            return None;
        }
        let stride = u32::from_le_bytes(data[6..10].try_into().unwrap());
        if stride == 0 {
            return None;
        }
        let mut pos = 10usize;
        let mut plane_meta = Vec::with_capacity(n_planes);
        for _ in 0..n_planes {
            if pos + 8 > data.len() {
                return None;
            }
            let n_blocks = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            let n_entries = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
            pos += 8;
            if n_blocks == 0 {
                return None;
            }
            // Entry count must exactly match the stride-derived chunk count.
            if n_entries as u64 != (n_blocks as u64).div_ceil(stride as u64) {
                return None;
            }
            plane_meta.push((n_blocks, n_entries as usize));
        }
        let mut planes = Vec::with_capacity(n_planes);
        for &(n_blocks, n_entries) in &plane_meta {
            if pos + 7 * n_entries + 4 > data.len() {
                return None;
            }
            let mut entries = Vec::with_capacity(n_entries);
            let mut prev_off = 0u32;
            for i in 0..n_entries {
                let byte_off = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                let dc_pred = i16::from_le_bytes(data[pos + 4..pos + 6].try_into().unwrap());
                let bit_off = data[pos + 6];
                pos += 7;
                if bit_off > 7 {
                    return None;
                }
                // Offsets must be strictly increasing within a plane.
                if i > 0 && byte_off <= prev_off {
                    return None;
                }
                prev_off = byte_off;
                entries.push(BlockRestart {
                    byte_off,
                    dc_pred,
                    bit_off,
                });
            }
            planes.push(PlaneIndex { n_blocks, entries });
        }
        if pos + 4 != data.len() {
            return None;
        }
        let want = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        if crc32(&data[..pos]) != want {
            return None;
        }
        Some(BgIndex { stride, planes })
    }
}

// ---------------------------------------------------------------------------
// Random-access index acceptance tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod random_access_tests {
    use super::*;
    use crate::block::Block;
    use crate::blockizer::Blockizer;
    use crate::colorspace;
    use crate::dct;
    use crate::encoder;
    use std::time::Instant;

    // The C SIMD quantize/dequantize (c/quant.c) is not linkable from
    // `cargo test`, so the tests use these local equivalents. Dequantize is
    // bit-exact with the C scalar path; quantize mirrors its float-division
    // semantics. Both the seek and the from-the-start decode paths below use
    // the same functions, so pixel-exactness is checked on equal footing.
    fn quantize_local(block: &mut Block, table: &[i16; 64]) {
        for i in 0..64 {
            block.data[i] = ((block.data[i] as f32) / (table[i] as f32)).round() as i16;
        }
    }

    fn dequantize_local(block: &mut Block, table: &[i16; 64]) {
        for i in 0..64 {
            let v = block.data[i] as i32 * table[i] as i32;
            block.data[i] = v.clamp(-32768, 32767) as i16;
        }
    }

    const W: usize = 256;
    const H: usize = 256;
    const QUALITY: u8 = 50;

    // -- deterministic sample material generators (no external assets) -------

    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Material 1: smooth gradients with a mild sinusoidal texture.
    fn gen_gradient() -> Vec<u8> {
        let mut img = vec![0u8; W * H * 3];
        for y in 0..H {
            for x in 0..W {
                let fx = x as f32;
                let fy = y as f32;
                let i = (y * W + x) * 3;
                let r = (x * 255 / (W - 1)) as f32 + 28.0 * (fx * 0.31).sin() * (fy * 0.23).cos();
                let g = (y * 255 / (H - 1)) as f32 + 22.0 * ((fx + fy) * 0.19).sin();
                let b = ((x + y) * 255 / (W + H - 2)) as f32
                    + 26.0 * (fy * 0.37).sin() * (fx * 0.29).cos();
                img[i] = r.clamp(0.0, 255.0) as u8;
                img[i + 1] = g.clamp(0.0, 255.0) as u8;
                img[i + 2] = b.clamp(0.0, 255.0) as u8;
            }
        }
        img
    }

    /// Material 2: uniform random noise (worst-case entropy).
    fn gen_noise() -> Vec<u8> {
        let mut img = vec![0u8; W * H * 3];
        let mut st = 0x1234_5678_9abc_def0u64;
        for p in img.iter_mut() {
            *p = (xorshift(&mut st) >> 32) as u8;
        }
        img
    }

    /// Material 3: hard edges — checkerboard, stripes, solid rectangles.
    fn gen_edges() -> Vec<u8> {
        let mut img = vec![0u8; W * H * 3];
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) * 3;
                let checker = ((x / 8) + (y / 8)) % 2 == 0;
                let (mut r, mut g, mut b) = if checker {
                    (255u8, 255u8, 255u8)
                } else {
                    (0u8, 0u8, 0u8)
                };
                if (x / 4) % 2 == 0 && y >= H / 2 {
                    r = 200;
                }
                if x >= W / 3 && x < W / 3 + 24 && y >= H / 3 && y < H / 3 + 24 {
                    g = 220;
                    b = 30;
                }
                img[i] = r;
                img[i + 1] = g;
                img[i + 2] = b;
            }
        }
        img
    }

    /// Material 4: photo-like RGBA — gradients + sinusoidal texture + shapes.
    fn gen_photo_rgba() -> Vec<u8> {
        let mut img = vec![0u8; W * H * 4];
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) * 4;
                let fx = x as f32;
                let fy = y as f32;
                let tex = (fx * 0.35).sin() * (fy * 0.22).cos() * 40.0;
                let cx = x as i32 - (W / 2) as i32;
                let cy = y as i32 - (H / 2) as i32;
                let in_disc = cx * cx + cy * cy < 30 * 30;
                let r = (x * 255 / (W - 1)) as f32 + tex;
                let g = (y * 255 / (H - 1)) as f32 + tex * 0.5;
                let b = (128.0 + 60.0 * ((fx + fy) * 0.05).sin()) as f32;
                img[i] = if in_disc { 240 } else { r.clamp(0.0, 255.0) as u8 };
                img[i + 1] = if in_disc { 90 } else { g.clamp(0.0, 255.0) as u8 };
                img[i + 2] = if in_disc { 60 } else { b.clamp(0.0, 255.0) as u8 };
                img[i + 3] = (255 - (x * 200 / (W - 1)) as i32).clamp(30, 255) as u8;
            }
        }
        img
    }

    // -- encode helpers -------------------------------------------------------

    struct PlaneMeta {
        pw: usize,
        ph: usize,
        is_chroma: bool,
        file_pos: usize,
    }

    struct EncodedMaterial {
        name: &'static str,
        channels: usize,
        stream: Vec<u8>,
        index: BgIndex,
        metas: Vec<PlaneMeta>,
    }

    /// Encode with the same pipeline the production encoder uses (colorspace,
    /// blockizer, DCT, quantize, clamp, Huffman) but through the indexed plane
    /// encoder so every block's restart point is recorded at encode time.
    fn encode_indexed(name: &'static str, pixels: &[u8], channels: usize) -> EncodedMaterial {
        let (w, h) = (W, H);
        let cw = (w + 1) / 2;
        let ch = (h + 1) / 2;
        let luma_q = encoder::quant_table_for_quality_perceptual_v4(QUALITY);
        let chroma_q = encoder::chroma_quant_table_for_quality_perceptual_v4(QUALITY);

        let mut plane_src: Vec<(Vec<u8>, usize, usize, bool)> = Vec::new();
        let version: u8;
        if channels == 3 {
            let (y, cb, cr) = colorspace::rgb_to_ycbcr420(pixels, w, h);
            plane_src.push((y, w, h, false));
            plane_src.push((cb, cw, ch, true));
            plane_src.push((cr, cw, ch, true));
            version = 18;
        } else {
            let (y, cb, cr, a) = colorspace::rgba_to_ycbcr420a(pixels, w, h);
            plane_src.push((y, w, h, false));
            plane_src.push((cb, cw, ch, true));
            plane_src.push((cr, cw, ch, true));
            plane_src.push((a, w, h, false));
            version = 19;
        }

        let mut stream = Vec::new();
        stream.extend_from_slice(b"BG");
        stream.push(version);
        stream.extend_from_slice(&(w as u32).to_le_bytes());
        stream.extend_from_slice(&(h as u32).to_le_bytes());
        stream.push(QUALITY);

        let mut per_plane: Vec<(usize, Vec<BlockRestart>)> = Vec::new();
        let mut metas = Vec::new();
        for (pix, pw, ph, is_chroma) in &plane_src {
            let table = if *is_chroma { &chroma_q } else { &luma_q };
            let mut blocks = Blockizer::new(*pw, *ph).generate_blocks(pix);
            for b in blocks.iter_mut() {
                dct::dct(b);
                quantize_local(b, table);
                clamp_block_jpeg_coeffs(b);
            }
            let (plane_buf, restarts) =
                encode_plane_with_profile_indexed(&blocks, *is_chroma, *is_chroma, true);
            let file_pos = stream.len();
            stream.extend_from_slice(&plane_buf);
            per_plane.push((file_pos, restarts));
            metas.push(PlaneMeta {
                pw: *pw,
                ph: *ph,
                is_chroma: *is_chroma,
                file_pos,
            });
        }

        // Budget: index must stay <= 2% of the bitstream.
        let index = BgIndex::build(&per_plane, stream.len() / 50);
        EncodedMaterial {
            name,
            channels,
            stream,
            index,
            metas,
        }
    }

    fn write_block_to_plane(block: &Block, plane: &mut [u8], w: usize, h: usize, idx: usize) {
        let bw = (w + 7) / 8;
        let by = (idx / bw) * 8;
        let bx = (idx % bw) * 8;
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

    /// Decode every chunk of every plane through the index and reassemble pixels.
    fn seek_decode_all(m: &EncodedMaterial) -> Vec<u8> {
        let luma_q = encoder::quant_table_for_quality_perceptual_v4(QUALITY);
        let chroma_q = encoder::chroma_quant_table_for_quality_perceptual_v4(QUALITY);
        let mut planes: Vec<Vec<u8>> = m
            .metas
            .iter()
            .map(|mt| vec![0u8; mt.pw * mt.ph])
            .collect();
        for (pi, mt) in m.metas.iter().enumerate() {
            let table = if mt.is_chroma { &chroma_q } else { &luma_q };
            let n_chunks = m.index.planes[pi].entries.len();
            for ci in 0..n_chunks {
                let (restart, first, len) = m.index.locate(pi, ci * m.index.stride as usize).unwrap();
                let mut blocks =
                    decode_blocks_at(&m.stream, &restart, len, mt.is_chroma, mt.is_chroma, true)
                        .expect("seek decode failed");
                for (k, b) in blocks.iter_mut().enumerate() {
                    dequantize_local(b, table);
                    dct::idct(b);
                    write_block_to_plane(b, &mut planes[pi], mt.pw, mt.ph, first + k);
                }
            }
        }
        planes_to_pixels(m.channels, &planes)
    }

    /// From-the-start decode of the whole stream: sequential Huffman decode of
    /// every plane (the decoder's entropy path) plus the same post-processing.
    fn full_decode(m: &EncodedMaterial) -> Vec<u8> {
        let luma_q = encoder::quant_table_for_quality_perceptual_v4(QUALITY);
        let chroma_q = encoder::chroma_quant_table_for_quality_perceptual_v4(QUALITY);
        let mut planes: Vec<Vec<u8>> = m
            .metas
            .iter()
            .map(|mt| vec![0u8; mt.pw * mt.ph])
            .collect();
        for (pi, mt) in m.metas.iter().enumerate() {
            let table = if mt.is_chroma { &chroma_q } else { &luma_q };
            let n_blocks = m.index.planes[pi].n_blocks as usize;
            let (mut blocks, _) = decode_plane_with_profile(
                &m.stream,
                mt.file_pos,
                n_blocks,
                mt.is_chroma,
                mt.is_chroma,
                true,
            )
            .expect("full plane decode failed");
            for (idx, b) in blocks.iter_mut().enumerate() {
                dequantize_local(b, table);
                dct::idct(b);
                write_block_to_plane(b, &mut planes[pi], mt.pw, mt.ph, idx);
            }
        }
        planes_to_pixels(m.channels, &planes)
    }

    fn planes_to_pixels(channels: usize, planes: &[Vec<u8>]) -> Vec<u8> {
        let (w, h) = (W, H);
        let mut out = vec![0u8; w * h * channels];
        if channels == 3
        {
            colorspace::ycbcr420_to_rgb(&planes[0], &planes[1], &planes[2], w, h, &mut out);
        } else {
            colorspace::ycbcr420a_to_rgba(
                &planes[0],
                &planes[1],
                &planes[2],
                &planes[3],
                w,
                h,
                &mut out,
            );
        }
        out
    }

    // -- tests ----------------------------------------------------------------

    /// The indexed encoder must produce a bitstream byte-identical to the
    /// classic one (index recording is observation-only).
    #[test]
    fn indexed_encode_matches_classic_bitstream() {
        let mut blocks = Vec::new();
        let mut st = 0xdead_beefu64;
        for _ in 0..50 {
            let mut b = Block::new();
            for v in b.data.iter_mut() {
                *v = ((xorshift(&mut st) >> 20) as i16).clamp(-1023, 1023);
            }
            blocks.push(b);
        }
        for &(is_chroma, chroma_ac, dc_delta) in
            &[(false, false, true), (true, true, true), (false, false, false)]
        {
            let classic = encode_plane_with_profile(&blocks, is_chroma, chroma_ac, dc_delta);
            let (indexed, restarts) =
                encode_plane_with_profile_indexed(&blocks, is_chroma, chroma_ac, dc_delta);
            assert_eq!(classic, indexed);
            assert_eq!(restarts.len(), blocks.len());
        }
    }

    /// Corrupt indexes must be rejected by `BgIndex::parse`.
    #[test]
    fn index_corruption_is_detected() {
        let m = encode_indexed("noise", &gen_noise(), 3);
        let blob = m.index.serialize();
        let parsed = BgIndex::parse(&blob).expect("valid index must parse");
        assert_eq!(parsed, m.index);

        // Flip every byte one at a time: each single-byte corruption must be caught.
        for i in 0..blob.len() {
            let mut bad = blob.clone();
            bad[i] ^= 0x5A;
            assert!(
                BgIndex::parse(&bad).is_none(),
                "corruption at byte {i} went undetected"
            );
        }
        // Truncation must be caught.
        for cut in [0, 1, 5, blob.len() / 2, blob.len() - 1] {
            assert!(BgIndex::parse(&blob[..cut]).is_none());
        }
        // Garbage must be caught.
        assert!(BgIndex::parse(&vec![0u8; 64]).is_none());
    }

    /// Full acceptance: four material types, index <= 2% of stream, seek decode
    /// pixel-exact vs full decode, seek latency far below full decode.
    #[test]
    fn random_access_acceptance() {
        let materials = vec![
            encode_indexed("gradient-rgb", &gen_gradient(), 3),
            encode_indexed("noise-rgb", &gen_noise(), 3),
            encode_indexed("edges-rgb", &gen_edges(), 3),
            encode_indexed("photo-rgba", &gen_photo_rgba(), 4),
        ];

        println!("\n==== random-access index acceptance ({}x{}, q{}) ====", W, H, QUALITY);
        println!(
            "{:<14} {:>9} {:>8} {:>7} {:>7} {:>7} {:>11} {:>11} {:>8}",
            "material", "stream B", "index B", "ratio%", "stride", "chunks",
            "full us", "seek us", "speedup"
        );

        for m in &materials {
            let blob = m.index.serialize();
            let ratio = blob.len() as f64 * 100.0 / m.stream.len() as f64;
            assert!(
                blob.len() * 50 <= m.stream.len(),
                "{}: index {} B exceeds 2% of stream {} B",
                m.name,
                blob.len(),
                m.stream.len()
            );

            // Index round-trips through its serialized form; everything below
            // runs against the *parsed* index, as a real consumer would.
            let parsed = BgIndex::parse(&blob).expect("index must parse");
            assert_eq!(parsed, m.index);
            let m = &EncodedMaterial {
                name: m.name,
                channels: m.channels,
                stream: m.stream.clone(),
                index: parsed,
                metas: m
                    .metas
                    .iter()
                    .map(|p| PlaneMeta {
                        pw: p.pw,
                        ph: p.ph,
                        is_chroma: p.is_chroma,
                        file_pos: p.file_pos,
                    })
                    .collect(),
            };

            // 1) Coefficient-level: every seek-decoded chunk matches the
            //    sequential Huffman decode exactly, block for block.
            for (pi, mt) in m.metas.iter().enumerate() {
                let n_blocks = m.index.planes[pi].n_blocks as usize;
                let (ref_blocks, _) = decode_plane_with_profile(
                    &m.stream,
                    mt.file_pos,
                    n_blocks,
                    mt.is_chroma,
                    mt.is_chroma,
                    true,
                )
                .expect("reference plane decode failed");
                let stride = m.index.stride as usize;
                for ci in 0..m.index.planes[pi].entries.len() {
                    let (restart, first, len) =
                        m.index.locate(pi, ci * stride).expect("locate failed");
                    let got = decode_blocks_at(
                        &m.stream,
                        &restart,
                        len,
                        mt.is_chroma,
                        mt.is_chroma,
                        true,
                    )
                    .expect("seek decode failed");
                    for (k, b) in got.iter().enumerate() {
                        assert_eq!(
                            b.data, ref_blocks[first + k].data,
                            "{}: plane {pi} block {} mismatch",
                            m.name,
                            first + k
                        );
                    }
                }
            }

            // 2) Pixel-level: reassembled image from independent chunk decodes
            //    is identical to a full decode.
            let px_full = full_decode(m);
            let px_seek = seek_decode_all(m);
            assert_eq!(
                px_full, px_seek,
                "{}: seek-decoded pixels differ from full decode",
                m.name
            );

            // 3) Timing: jumping to one chunk vs decoding the whole stream.
            let full_iters = 50;
            let t0 = Instant::now();
            for _ in 0..full_iters {
                std::hint::black_box(full_decode(m));
            }
            let full_us = t0.elapsed().as_micros() as f64 / full_iters as f64;

            // Seek to the middle chunk of the luma plane and decode it
            // (incl. dequant + IDCT), index already parsed.
            let mid = m.index.planes[0].entries.len() / 2;
            let (restart, _first, len) =
                m.index.locate(0, mid * m.index.stride as usize).unwrap();
            let luma_q = encoder::quant_table_for_quality_perceptual_v4(QUALITY);
            let seek_iters = 500;
            let t0 = Instant::now();
            for _ in 0..seek_iters {
                let mut blocks = decode_blocks_at(&m.stream, &restart, len, false, false, true)
                    .expect("seek decode failed");
                for b in blocks.iter_mut() {
                    dequantize_local(b, &luma_q);
                    dct::idct(b);
                }
                std::hint::black_box(&blocks);
            }
            let seek_us = t0.elapsed().as_micros() as f64 / seek_iters as f64;

            let n_chunks: usize = m.index.planes.iter().map(|p| p.entries.len()).sum();
            println!(
                "{:<14} {:>9} {:>8} {:>6.2} {:>7} {:>7} {:>11.1} {:>11.2} {:>7.1}x",
                m.name,
                m.stream.len(),
                blob.len(),
                ratio,
                m.index.stride,
                n_chunks,
                full_us,
                seek_us,
                full_us / seek_us
            );
            assert!(
                seek_us * 4.0 < full_us,
                "{}: seek decode not significantly faster ({seek_us:.1} us vs {full_us:.1} us)",
                m.name
            );
        }
        println!("all materials: pixel-exact seek decode, index <= 2% of stream, corruption detected\n");
    }
}
