use crate::block::Block;
use crate::huffman::{
    clamp_block_jpeg_coeffs, decode_plane, decode_plane_with_ac, decode_plane_with_profile, encode_plane,
    encode_plane_with_ac, encode_plane_with_period, encode_plane_with_profile,
};
use crate::zigzag::ZIGZAG;

fn make_block(vals: &[(usize, i16)]) -> Block {
    let mut b = Block::new();
    for &(zi, v) in vals {
        b.data[ZIGZAG[zi]] = v;
    }
    b
}

#[test]
fn huffman_roundtrip_single_block() {
    let block = make_block(&[(0, 100), (1, 50), (2, -30), (5, 10)]);
    let encoded = encode_plane(&[block], false);
    let (decoded, _) = decode_plane(&encoded, 0, 1, false).expect("decode failed");
    assert_eq!(decoded[0].data, block.data, "roundtrip mismatch");
}

#[test]
fn huffman_roundtrip_all_zeros() {
    let block = Block::new();
    let encoded = encode_plane(&[block], false);
    let (decoded, _) = decode_plane(&encoded, 0, 1, false).expect("decode failed");
    assert_eq!(decoded[0].data, block.data);
}

#[test]
fn huffman_roundtrip_multiple_blocks() {
    let blocks: Vec<Block> = (0..16i16)
        .map(|i| make_block(&[(0, i * 10), (1, i * 3 - 20), (3, i - 5)]))
        .collect();
    let encoded = encode_plane(&blocks, false);
    let (decoded, _) = decode_plane(&encoded, 0, 16, false).expect("decode failed");
    for (i, (orig, dec)) in blocks.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(orig.data, dec.data, "block {i} mismatch");
    }
}

#[test]
fn huffman_roundtrip_chroma() {
    let block = make_block(&[(0, 20), (1, -15), (4, 8)]);
    let encoded = encode_plane(&[block], true);
    let (decoded, _) = decode_plane(&encoded, 0, 1, true).expect("chroma decode failed");
    assert_eq!(decoded[0].data, block.data);
}

#[test]
fn huffman_roundtrip_chroma_ac_table_v2() {
    let block = make_block(&[(0, 20), (1, -15), (4, 8), (9, -3), (10, 2)]);
    let encoded = encode_plane_with_ac(&[block], true, true);
    let (decoded, _) = decode_plane_with_ac(&encoded, 0, 1, true, true).expect("chroma-v2 decode failed");
    assert_eq!(decoded[0].data, block.data);
}

#[test]
fn huffman_long_zero_run_rs_symbol() {
    let mut b = Block::new();
    b.data[ZIGZAG[16]] = 1;
    let encoded = encode_plane(&[b], false);
    let (dec, _) = decode_plane(&encoded, 0, 1, false).expect("decode");
    assert_eq!(dec[0].data, b.data);
}

#[test]
fn huffman_roundtrip_after_jpeg_clamp() {
    let mut b = Block::new();
    b.data[ZIGZAG[1]] = 1500;
    clamp_block_jpeg_coeffs(&mut b);
    assert_eq!(b.data[ZIGZAG[1]], 1023);
    let encoded = encode_plane(&[b], false);
    let (dec, _) = decode_plane(&encoded, 0, 1, false).expect("decode");
    assert_eq!(dec[0].data, b.data);
}

#[test]
fn huffman_consecutive_planes() {
    let y_blocks: Vec<Block> = (0..4).map(|i| make_block(&[(0, i * 20), (1, i * 5)])).collect();
    let cb_blocks: Vec<Block> = (0..1).map(|i| make_block(&[(0, i * 10 + 5)])).collect();

    let y_buf = encode_plane(&y_blocks, false);
    let cb_buf = encode_plane(&cb_blocks, true);

    let mut combined = y_buf.clone();
    combined.extend_from_slice(&cb_buf);

    let (y_dec, y_end) = decode_plane(&combined, 0, 4, false).expect("Y decode");
    let (cb_dec, _) = decode_plane(&combined, y_end, 1, true).expect("Cb decode");

    for (i, (o, d)) in y_blocks.iter().zip(y_dec.iter()).enumerate() {
        assert_eq!(o.data, d.data, "Y block {i}");
    }
    for (i, (o, d)) in cb_blocks.iter().zip(cb_dec.iter()).enumerate() {
        assert_eq!(o.data, d.data, "Cb block {i}");
    }
}

#[test]
fn huffman_roundtrip_stress_random_blocks() {
    // Deterministic LCG to avoid external rand dependency.
    fn next_u32(state: &mut u64) -> u32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 32) as u32
    }

    let mut seed = 0x1234_5678_9abc_def0u64;
    let n_blocks = 4096usize;
    let mut blocks = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let mut b = Block::new();
        // Dense, wide-range random coefficients to stress symbol coverage.
        b.data[ZIGZAG[0]] = (next_u32(&mut seed) % 4095) as i16 - 2047;
        for zi in 1..64 {
            let r = next_u32(&mut seed) % 100;
            if r < 65 {
                b.data[ZIGZAG[zi]] = 0;
            } else {
                b.data[ZIGZAG[zi]] = (next_u32(&mut seed) % 2047) as i16 - 1023;
            }
        }
        clamp_block_jpeg_coeffs(&mut b);
        blocks.push(b);
    }

    let encoded = encode_plane(&blocks, false);
    let (decoded, _) = decode_plane(&encoded, 0, n_blocks, false).expect("stress decode failed");
    for (i, (orig, dec)) in blocks.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(orig.data, dec.data, "stress mismatch in block {i}");
    }
}

#[test]
fn huffman_roundtrip_dc_delta_profile() {
    let blocks: Vec<Block> = (0..64i16)
        .map(|i| make_block(&[(0, i * 13 - 250), (1, i % 7 - 3), (6, (i % 5) - 2)]))
        .collect();
    let encoded = encode_plane_with_profile(&blocks, false, false, true);
    let (decoded, _) =
        decode_plane_with_profile(&encoded, 0, blocks.len(), false, false, true).expect("dc-delta decode");
    for (i, (orig, dec)) in blocks.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(orig.data, dec.data, "dc-delta mismatch in block {i}");
    }
}

// ---------------------------------------------------------------------------
// Adaptive arithmetic coder tests
// ---------------------------------------------------------------------------

fn lcg_blocks(n: usize, seed: &mut u64) -> Vec<Block> {
    fn next(state: &mut u64) -> u32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (*state >> 32) as u32
    }
    let mut blocks = Vec::with_capacity(n);
    for _ in 0..n {
        let mut b = Block::new();
        b.data[ZIGZAG[0]] = (next(seed) % 4095) as i16 - 2047;
        for zi in 1..64 {
            if next(seed) % 100 < 65 {
                b.data[ZIGZAG[zi]] = 0;
            } else {
                b.data[ZIGZAG[zi]] = (next(seed) % 2047) as i16 - 1023;
            }
        }
        blocks.push(b);
    }
    blocks
}

#[test]
fn arith_roundtrip_period_1() {
    let mut seed = 0xdead_beef_cafe_1234u64;
    let blocks = lcg_blocks(512, &mut seed);
    let encoded = encode_plane_with_period(&blocks, false, false, true, 1);
    let (decoded, _) = decode_plane_with_profile(&encoded, 0, blocks.len(), false, false, true)
        .expect("decode");
    for (i, (o, d)) in blocks.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.data, d.data, "period 1 mismatch block {i}");
    }
}

#[test]
fn arith_roundtrip_period_0_static() {
    let mut seed = 0x0bad_1dea_0bad_1deau64;
    let blocks = lcg_blocks(512, &mut seed);
    let encoded = encode_plane_with_period(&blocks, false, false, true, 0);
    let (decoded, _) = decode_plane_with_profile(&encoded, 0, blocks.len(), false, false, true)
        .expect("decode");
    for (i, (o, d)) in blocks.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.data, d.data, "static-model mismatch block {i}");
    }
}

#[test]
fn arith_roundtrip_period_7() {
    let mut seed = 0x1234_4321_5678_8765u64;
    let blocks = lcg_blocks(1024, &mut seed);
    let encoded = encode_plane_with_period(&blocks, false, false, true, 7);
    let (decoded, _) = decode_plane_with_profile(&encoded, 0, blocks.len(), false, false, true)
        .expect("decode");
    for (i, (o, d)) in blocks.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(o.data, d.data, "period 7 mismatch block {i}");
    }
}

#[test]
fn arith_roundtrip_all_zero_and_extremes() {
    for period in [0u8, 1, 3, 127] {
        let mut blocks = vec![Block::new(); 4];
        let mut extreme = Block::new();
        extreme.data[ZIGZAG[0]] = i16::MIN + 1;
        extreme.data[ZIGZAG[1]] = i16::MAX;
        extreme.data[ZIGZAG[63]] = -1;
        blocks.push(extreme);
        let encoded = encode_plane_with_period(&blocks, false, false, true, period);
        let (decoded, _) = decode_plane_with_profile(&encoded, 0, blocks.len(), false, false, true)
            .expect("decode");
        for (i, (o, d)) in blocks.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(o.data, d.data, "extreme mismatch period {period} block {i}");
        }
    }
}

#[test]
fn arith_payload_has_magic_and_period() {
    let block = make_block(&[(0, 7), (2, -3)]);
    let encoded = encode_plane_with_period(&[block], false, false, false, 5);
    assert_eq!(&encoded[4..8], b"ARIT");
    assert_eq!(encoded[8], 1); // codec version
    assert_eq!(encoded[9] & 0x7F, 5); // update period
}

#[test]
fn legacy_huffman_payload_still_decodes() {
    let mut seed = 0xfeed_face_0000_1111u64;
    let mut blocks = lcg_blocks(300, &mut seed);
    // The legacy Huffman tables only cover DC category ≤ 11 (|delta| ≤ 2047) and AC ≤ 1023.
    // Keep DC within ±1023 so any delta between consecutive blocks stays representable.
    for b in blocks.iter_mut() {
        clamp_block_jpeg_coeffs(b);
        b.data[ZIGZAG[0]] = b.data[ZIGZAG[0]].clamp(-1023, 1023);
    }
    for &(chroma, dc_delta) in &[(false, false), (false, true), (true, true)] {
        let legacy = crate::huffman::legacy_encode_plane_with_profile(&blocks, chroma, chroma, dc_delta);
        assert_ne!(&legacy[4..8], b"ARIT", "legacy payload should not carry the magic");
        let (decoded, _) = decode_plane_with_profile(&legacy, 0, blocks.len(), chroma, chroma, dc_delta)
            .expect("legacy decode");
        for (i, (o, d)) in blocks.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(o.data, d.data, "legacy mismatch chroma={chroma} dc={dc_delta} block {i}");
        }
    }
}
