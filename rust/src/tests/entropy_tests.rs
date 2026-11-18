//! Tests for the pluggable entropy-coding interface and its backends.
//!
//! Acceptance covered here:
//!   - the extracted Huffman backend is byte-identical to the legacy
//!     hardcoded path (lossless extraction);
//!   - the zero-order arithmetic backend round-trips losslessly;
//!   - both backends decode to identical pixels on full images;
//!   - `entropy_bench_100mb` (ignored, run explicitly) measures ratio,
//!     encode/decode time and interface overhead on a 100 MiB input.

use crate::arith::{decode_plane_arith, ArithEncoder};
use crate::block::Block;
use crate::encoder::{self, EntropyKind};
use crate::entropy_coder::{encode_plane_walk, EntropyEncoder};
use crate::huffman::{self, HuffmanEncoder};
use crate::zigzag::ZIGZAG;

// ---------------------------------------------------------------------------
// Deterministic test data
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Blocks with JPEG-clamped coefficients, exercising EOB, ZRL (long zero
/// runs), all-zero blocks, negative values and max categories.
/// DCs follow a bounded random walk (like real DCT DCs), so DC deltas stay
/// inside the category-11 tables.
fn make_test_blocks(n: usize, seed: u64) -> Vec<Block> {
    let mut rng = Lcg(seed);
    let mut blocks = Vec::with_capacity(n);
    let mut dc_walk: i16 = 0;
    for bi in 0..n {
        let mut b = Block::new();
        match bi % 4 {
            0 => {
                // sparse: DC + a few AC
                dc_walk = (dc_walk + (rng.below(511) as i16) - 255).clamp(-2047, 2047);
                b.data[ZIGZAG[0]] = dc_walk;
                for _ in 0..rng.below(6) {
                    let zi = 1 + rng.below(63) as usize;
                    b.data[ZIGZAG[zi]] = (rng.below(2046) as i16) - 1023;
                }
            }
            1 => {
                // dense: every coefficient non-zero (max categories)
                for zi in 0..64 {
                    b.data[ZIGZAG[zi]] = if zi == 0 {
                        dc_walk = (dc_walk + (rng.below(511) as i16) - 255).clamp(-2047, 2047);
                        dc_walk
                    } else {
                        (rng.below(2046) as i16) - 1023
                    };
                }
            }
            2 => {
                // long zero run in the middle (forces ZRL symbols)
                dc_walk = (dc_walk + (rng.below(201) as i16) - 100).clamp(-2047, 2047);
                b.data[ZIGZAG[0]] = dc_walk;
                b.data[ZIGZAG[1]] = 7;
                b.data[ZIGZAG[62]] = -3;
            }
            _ => {
                // all-zero AC (pure EOB)
                dc_walk = (dc_walk + (rng.below(511) as i16) - 255).clamp(-2047, 2047);
                b.data[ZIGZAG[0]] = dc_walk;
            }
        }
        blocks.push(b);
    }
    blocks
}

const PROFILES: [(bool, bool, bool); 6] = [
    // (is_chroma, use_chroma_ac, use_dc_delta) — every combo the encoder uses
    (false, false, false),
    (false, false, true),
    (true, false, false),
    (true, true, false),
    (true, true, true),
    (false, false, true),
];

fn encode_iface_huffman(blocks: &[Block], is_chroma: bool, use_chroma_ac: bool, use_dc_delta: bool) -> Vec<u8> {
    let mut enc = HuffmanEncoder::new();
    enc.init();
    encode_plane_walk(&mut enc, blocks, is_chroma, use_chroma_ac, use_dc_delta);
    enc.finish()
}

fn encode_iface_arith(blocks: &[Block], is_chroma: bool, use_chroma_ac: bool, use_dc_delta: bool) -> Vec<u8> {
    let mut enc = ArithEncoder::new();
    enc.init();
    encode_plane_walk(&mut enc, blocks, is_chroma, use_chroma_ac, use_dc_delta);
    enc.finish()
}

// ---------------------------------------------------------------------------
// 1. Extraction is lossless: interface Huffman == legacy hardcoded, byte-exact
// ---------------------------------------------------------------------------

#[test]
fn iface_huffman_matches_legacy_byte_exact() {
    let blocks = make_test_blocks(300, 0xB16B00B5);
    for &(is_chroma, use_chroma_ac, use_dc_delta) in &PROFILES {
        let legacy = huffman::encode_plane_with_profile(&blocks, is_chroma, use_chroma_ac, use_dc_delta);
        let iface = encode_iface_huffman(&blocks, is_chroma, use_chroma_ac, use_dc_delta);
        assert_eq!(
            legacy, iface,
            "Huffman backend via interface differs from legacy path \
             (is_chroma={is_chroma}, use_chroma_ac={use_chroma_ac}, use_dc_delta={use_dc_delta})"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Arithmetic backend: plane-level round-trip
// ---------------------------------------------------------------------------

#[test]
fn arith_plane_roundtrip() {
    let blocks = make_test_blocks(300, 0xA117);
    for &(is_chroma, use_chroma_ac, use_dc_delta) in &PROFILES {
        let encoded = encode_iface_arith(&blocks, is_chroma, use_chroma_ac, use_dc_delta);
        let (decoded, _pos) =
            decode_plane_arith(&encoded, 0, blocks.len(), is_chroma, use_chroma_ac, use_dc_delta)
                .expect("arith decode failed");
        assert_eq!(decoded.len(), blocks.len());
        for (i, (a, b)) in blocks.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(a.data, b.data, "block {i} mismatch (profile {is_chroma}/{use_chroma_ac}/{use_dc_delta})");
        }
    }
}

#[test]
fn arith_consecutive_planes() {
    // Two planes back-to-back: the length prefix must land the second decoder
    // exactly on the second plane boundary.
    let b1 = make_test_blocks(40, 1);
    let b2 = make_test_blocks(30, 2);
    let mut stream = encode_iface_arith(&b1, false, false, true);
    let p2 = stream.len();
    stream.extend_from_slice(&encode_iface_arith(&b2, true, true, true));
    let (d1, next) = decode_plane_arith(&stream, 0, b1.len(), false, false, true).unwrap();
    assert_eq!(next, p2);
    let (d2, _) = decode_plane_arith(&stream, next, b2.len(), true, true, true).unwrap();
    for (a, b) in b1.iter().zip(d1.iter()) {
        assert_eq!(a.data, b.data);
    }
    for (a, b) in b2.iter().zip(d2.iter()) {
        assert_eq!(a.data, b.data);
    }
}

// ---------------------------------------------------------------------------
// 3. Full-image: switch backends, bytes and decoded pixels
// ---------------------------------------------------------------------------

fn make_test_image_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut img = vec![0u8; w * h * 3];
    let mut rng = Lcg(42);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let n = (rng.below(16)) as u8;
            img[i] = ((x * 255 / w.max(1)) as u8).wrapping_add(n);
            img[i + 1] = ((y * 255 / h.max(1)) as u8).wrapping_add(n);
            img[i + 2] = (((x + y) * 127 / (w + h).max(1)) as u8).wrapping_add(n);
        }
    }
    img
}

fn encode_rgb_kind(img: &[u8], w: usize, h: usize, q: u8, kind: EntropyKind) -> Vec<u8> {
    let mut out = vec![0u8; img.len() * 2 + 4096];
    let mut pos: i32 = 0;
    encoder::encode_rgb_ycbcr_kind(img, w, h, q, &mut out, &mut pos, None, kind);
    out.truncate(pos as usize);
    out
}

fn encode_rgba_kind(img: &[u8], w: usize, h: usize, q: u8, kind: EntropyKind) -> Vec<u8> {
    let mut out = vec![0u8; img.len() * 2 + 4096];
    let mut pos: i32 = 0;
    encoder::encode_rgba_ycbcr_kind(img, w, h, q, &mut out, &mut pos, None, kind);
    out.truncate(pos as usize);
    out
}

fn decode_all(stream: &[u8], w: usize, h: usize, ch: usize) -> Vec<u8> {
    // The decoder writes through the whole out slice, so it must be exact-size.
    let mut out = vec![0u8; w * h * ch];
    let (mut dw, mut dh, mut dc) = (0u32, 0u32, 0u32);
    assert!(crate::decoder::decode(stream, &mut out, &mut dw, &mut dh, &mut dc, None), "decode failed");
    assert_eq!((dw as usize, dh as usize, dc as usize), (w, h, ch));
    out
}

#[test]
fn full_image_rgb_backends() {
    let (w, h) = (130usize, 101usize); // odd size exercises border blocks
    let img = make_test_image_rgb(w, h);

    let legacy = encode_rgb_kind(&img, w, h, 85, EntropyKind::Legacy);
    let huff = encode_rgb_kind(&img, w, h, 85, EntropyKind::Huffman);
    let arith = encode_rgb_kind(&img, w, h, 85, EntropyKind::Arith);

    // Lossless extraction: switching legacy -> huffman changes nothing.
    assert_eq!(legacy, huff, "interface Huffman stream differs from legacy");
    assert_eq!(huff[2], 18, "huffman stream must keep version 18");
    assert_eq!(arith[2], 20, "arith stream must use version 20");

    // Both backends decode to identical pixels (entropy coding is lossless).
    let px_h = decode_all(&huff, w, h, 3);
    let px_a = decode_all(&arith, w, h, 3);
    assert_eq!(px_h.len(), w * h * 3);
    assert_eq!(px_h, px_a, "huffman and arith decodes differ");
}

#[test]
fn full_image_rgba_backends() {
    let (w, h) = (96usize, 64usize);
    let rgb = make_test_image_rgb(w, h);
    let mut img = vec![0u8; w * h * 4];
    for px in 0..w * h {
        img[px * 4] = rgb[px * 3];
        img[px * 4 + 1] = rgb[px * 3 + 1];
        img[px * 4 + 2] = rgb[px * 3 + 2];
        img[px * 4 + 3] = (px % 251) as u8;
    }

    let legacy = encode_rgba_kind(&img, w, h, 90, EntropyKind::Legacy);
    let huff = encode_rgba_kind(&img, w, h, 90, EntropyKind::Huffman);
    let arith = encode_rgba_kind(&img, w, h, 90, EntropyKind::Arith);

    assert_eq!(legacy, huff);
    assert_eq!(huff[2], 19);
    assert_eq!(arith[2], 21);

    let px_h = decode_all(&huff, w, h, 4);
    let px_a = decode_all(&arith, w, h, 4);
    assert_eq!(px_h, px_a);
}

// ---------------------------------------------------------------------------
// 4. 100 MiB benchmark: ratio, time, interface overhead (run explicitly):
//    cargo test --release -- --ignored --nocapture entropy_bench_100mb
// ---------------------------------------------------------------------------

/// Backend that does nothing: measures the pure cost of driving the
/// interface (walk + per-symbol calls) with zero backend work.
struct NullBackend;

impl EntropyEncoder for NullBackend {
    fn init(&mut self) {}
    #[inline]
    fn encode_symbol(&mut self, class: crate::entropy_coder::SymClass, sym: u8) {
        std::hint::black_box((class, sym));
    }
    #[inline]
    fn encode_raw_bits(&mut self, bits: u16, n: u8) {
        std::hint::black_box((bits, n));
    }
    fn finish(&mut self) -> Vec<u8> {
        Vec::new()
    }
}

#[test]
#[ignore]
fn entropy_bench_100mb() {
    use std::time::Instant;

    // 100 MiB of RGB: 6144 x 5689 x 3 = 104,859,648 bytes.
    let (w, h) = (6144usize, 5689usize);
    let raw = w * h * 3;
    eprintln!("generating {} MiB test image {}x{}...", raw >> 20, w, h);
    let mut img = vec![0u8; raw];
    let mut rng = Lcg(7);
    for y in 0..h {
        let row = y * w * 3;
        for x in 0..w {
            let i = row + x * 3;
            let n = rng.below(24) as u8;
            img[i] = ((x / 24) as u8).wrapping_add(n);
            img[i + 1] = ((y / 24) as u8).wrapping_add(n);
            img[i + 2] = (((x + y) / 32) as u8).wrapping_add(n);
        }
    }

    let mut out = vec![0u8; raw * 2 + 65536];
    let q = 85u8;
    let mut streams: Vec<(EntropyKind, Vec<u8>, f64)> = Vec::new();

    // Min of several runs: robust against CPU frequency / rayon scheduling noise.
    fn bench_min_ms(mut f: impl FnMut() -> (), reps: usize) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t0 = Instant::now();
            f();
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        best
    }

    for kind in [EntropyKind::Legacy, EntropyKind::Huffman, EntropyKind::Arith] {
        let mut stream = Vec::new();
        let ms = bench_min_ms(
            || {
                let mut pos: i32 = 0;
                encoder::encode_rgb_ycbcr_kind(&img, w, h, q, &mut out, &mut pos, None, kind);
                stream = out[..pos as usize].to_vec();
            },
            3,
        );
        streams.push((kind, stream, ms));
    }

    let get = |k: EntropyKind| streams.iter().find(|s| s.0 == k).unwrap();
    let (t_legacy, t_huff, t_arith) = (get(EntropyKind::Legacy).2, get(EntropyKind::Huffman).2, get(EntropyKind::Arith).2);
    let (s_legacy, s_huff, s_arith) = (&get(EntropyKind::Legacy).1, &get(EntropyKind::Huffman).1, &get(EntropyKind::Arith).1);

    // Lossless extraction check on the 100 MiB input.
    assert_eq!(s_legacy, s_huff, "interface Huffman != legacy on 100 MiB input");

    // Decode both backend streams, verify pixel-identical output.
    let mut px_h = vec![0u8; raw];
    let mut px_a = vec![0u8; raw];
    let (mut dw, mut dh, mut dc) = (0u32, 0u32, 0u32);
    let t_dec_huff = bench_min_ms(
        || {
            assert!(crate::decoder::decode(s_huff, &mut px_h, &mut dw, &mut dh, &mut dc, None));
        },
        3,
    );
    let t_dec_arith = bench_min_ms(
        || {
            assert!(crate::decoder::decode(s_arith, &mut px_a, &mut dw, &mut dh, &mut dc, None));
        },
        3,
    );
    assert_eq!(px_h, px_a, "backend decodes differ on 100 MiB input");

    // Interface overhead, isolated at plane level on the real Y plane:
    // same blocks, same walk — legacy hardcoded vs interface+Huffman vs
    // interface+Null (pure interface cost).
    let (y, _cb, _cr) = crate::colorspace::rgb_to_ycbcr420(&img, w, h);
    let blockizer = crate::blockizer::Blockizer::new(w, h);
    let mut blocks = blockizer.generate_blocks(&y);
    let table = encoder::quant_table_for_quality_perceptual_v4(q);
    let thr = {
        // same sparsify thresholds as the encoder path (quality 85, luma)
        let mut t = [0i16; 64];
        for (zi, e) in t.iter_mut().enumerate().skip(1) {
            *e = if zi >= 56 { 4 } else if zi >= 40 { 3 } else if zi >= 24 { 2 } else { 1 };
        }
        t
    };
    for b in blocks.iter_mut() {
        crate::dct::dct(b);
        encoder::quantize(&mut b.data, &table);
        for zi in 1..64 {
            let z = ZIGZAG[zi];
            if b.data[z].abs() <= thr[zi] {
                b.data[z] = 0;
            }
        }
        huffman::clamp_block_jpeg_coeffs(b);
    }

    const REPS: usize = 9;
    let p_legacy = bench_min_ms(
        || {
            std::hint::black_box(huffman::encode_plane_with_profile(std::hint::black_box(&blocks), false, false, true));
        },
        REPS,
    );

    let p_iface = bench_min_ms(
        || {
            std::hint::black_box(encode_iface_huffman(std::hint::black_box(&blocks), false, false, true));
        },
        REPS,
    );

    let p_null = bench_min_ms(
        || {
            let mut enc = NullBackend;
            enc.init();
            encode_plane_walk(&mut enc, std::hint::black_box(&blocks), false, false, true);
            std::hint::black_box(enc.finish());
        },
        REPS,
    );

    let mib = raw as f64 / (1024.0 * 1024.0);
    let ratio = |len: usize| mib / (len as f64 / (1024.0 * 1024.0));
    let overhead = (t_huff - t_legacy) / t_legacy * 100.0;
    let p_overhead = (p_iface - p_legacy) / p_legacy * 100.0;

    eprintln!("\n================ entropy backend bench (100 MiB RGB, q=85) ================");
    eprintln!("input: {raw} bytes ({mib:.1} MiB), {w}x{h}");
    eprintln!("--- full image encode (incl. DCT/quant/colorspace) ---");
    eprintln!("legacy hardcoded : {:9.1} ms  {:>10} bytes  ratio {:.3}:1", t_legacy, s_legacy.len(), ratio(s_legacy.len()));
    eprintln!("iface  huffman   : {:9.1} ms  {:>10} bytes  ratio {:.3}:1", t_huff, s_huff.len(), ratio(s_huff.len()));
    eprintln!("iface  arith     : {:9.1} ms  {:>10} bytes  ratio {:.3}:1", t_arith, s_arith.len(), ratio(s_arith.len()));
    eprintln!("--- decode ---");
    eprintln!("huffman stream   : {t_dec_huff:9.1} ms");
    eprintln!("arith  stream    : {t_dec_arith:9.1} ms");
    eprintln!("--- interface overhead ---");
    eprintln!("full-encode overhead (iface-huffman vs legacy): {overhead:+.2}%");
    eprintln!("plane walk only  : legacy {p_legacy:.1} ms | iface {p_iface:.1} ms ({p_overhead:+.2}%) | null-backend {p_null:.1} ms");
    eprintln!("iface huffman == legacy bytes: {}", s_legacy == s_huff);
    eprintln!("huffman decode == arith decode (pixels): {}", px_h == px_a);
    eprintln!("===========================================================================");
}
