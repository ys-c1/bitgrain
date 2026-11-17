//! 高频量化表共享 / 下采样 —— 三段式方案实验（库内研究模块）。
//!
//! # 边界：为什么是实验容器而不是改 `.bg`
//!
//! 当前 `.bg` 格式**不随流存储量化表**：解码端按 header 里的 quality 字节用确定性
//! 公式重建表（见 `encoder::scale_quant_table*` / decoder 各版本分支）。本模块在
//! 一个独立实验容器（magic `BQT`）内模拟"逐块高频量化表"场景，评估题目三段：
//!
//! 1. 相邻 2×2 个 8×8 块共用一张表（tile 级共享）；
//! 2. 按块活跃度在**平滑 / 纹理两套共享表**之间选择（两套表内容可配）；
//! 3. 按率失真代价 `J = D + λ·R` 决定下采样粒度 g ∈ {1,2,4,8}，
//!    g=1（不共享）必须与原编码**逐字节一致**。
//!
//! 原编码成本模型：每块携带一张由该块 DCT 活跃度确定性自适应得到的 64 项 i16 表
//! （DC 保持基准表，AC 按活跃度整体缩放），表区随块数线性增长，对应
//! "高频量化表占体积不小"的前提。g≥2 时全图只存两套表 + 每块 1 bit 选择位图。
//!
//! # 约束遵守
//!
//! - 本模块是自包含新增文件，仅在 `lib.rs` 增加一行 `mod` 声明；
//!   不修改任何既有编码/解码代码、构建脚本与公共接口签名，`.bg` 输出不可能变化。
//! - 只用 crate 内已有纯 Rust 原语（参考 DCT/IDCT、JPEG 基准表、zigzag、Block）
//!   与 std；不引入第三方库、不访问网络；确定性运算，无 RNG。
//! - 所有对外项保持私有（模块不 `pub`），仅 `#[cfg(test)]` 单测使用，
//!   不进入公共 API；样例为 32×32 小型合成图，`cargo test` 一次跑完。

use crate::block::Block;
use crate::dct::{dct_reference, idct_reference};
use crate::encoder::quant_table_for_quality;
use crate::zigzag::ZIGZAG;

// ---------------------------------------------------------------------------
// 配置（共享粒度、两套表内容、率失真权重均可配）
// ---------------------------------------------------------------------------

/// 平滑表默认频率权重（百分比，zigzag 顺序）：高频压得更狠。
const DEFAULT_SMOOTH_PROFILE: [u8; 64] = [
    100, 102, 104, 106, 108, 112, 116, 122,
    102, 104, 106, 110, 114, 118, 124, 130,
    104, 106, 110, 114, 120, 126, 132, 138,
    106, 110, 114, 120, 126, 132, 138, 144,
    108, 114, 120, 126, 132, 138, 144, 150,
    112, 118, 126, 132, 138, 144, 150, 154,
    116, 124, 132, 138, 144, 150, 154, 158,
    122, 130, 138, 144, 150, 154, 158, 160,
];

/// 纹理表默认频率权重（百分比）：高频保留更多细节。
const DEFAULT_TEXTURE_PROFILE: [u8; 64] = [
    100, 100, 101, 102, 103, 104, 106, 108,
    100, 101, 102, 103, 104, 106, 108, 110,
    101, 102, 103, 104, 106, 108, 110, 112,
    102, 103, 104, 106, 108, 110, 112, 114,
    103, 104, 106, 108, 110, 112, 114, 116,
    104, 106, 108, 110, 112, 114, 116, 118,
    106, 108, 110, 112, 114, 116, 118, 120,
    108, 110, 112, 114, 116, 118, 120, 122,
];

#[derive(Clone, Debug)]
struct SharingConfig {
    /// 共享/下采样粒度：1=不共享（原编码），2=2×2 块，4=4×4 块，8=8×8 块。
    granularity: u8,
    /// 平滑/纹理分类阈值 = activity_bias(permil) × 全图平均活跃度 / 1000。
    activity_bias: u32,
    /// 平滑套表频率权重（百分比），表内容可配。
    smooth_profile: [u8; 64],
    /// 纹理套表频率权重（百分比），表内容可配。
    texture_profile: [u8; 64],
    /// 率失真权重 λ（千分比定点）：J = SSE + λ/1000 · R_bytes。
    rd_lambda_permil: u32,
}

impl Default for SharingConfig {
    fn default() -> Self {
        Self {
            granularity: 2,
            activity_bias: 1000,
            smooth_profile: DEFAULT_SMOOTH_PROFILE,
            texture_profile: DEFAULT_TEXTURE_PROFILE,
            rd_lambda_permil: 1000,
        }
    }
}

impl SharingConfig {
    fn with_granularity(mut self, g: u8) -> Self {
        assert!(matches!(g, 1 | 2 | 4 | 8), "granularity must be 1,2,4,8");
        self.granularity = g;
        self
    }
}

// ---------------------------------------------------------------------------
// 块原语：分块、参考 DCT、整数量化
// ---------------------------------------------------------------------------

fn blocks_wide(w: usize) -> usize { (w + 7) / 8 }
fn blocks_high(h: usize) -> usize { (h + 7) / 8 }

/// 8×8 分块（边缘像素复制，与 Blockizer 同构）+ crate 自带的纯 Rust 参考 DCT。
fn forward_blocks(image: &[u8], w: usize, h: usize) -> Vec<Block> {
    let bw = blocks_wide(w);
    let bh = blocks_high(h);
    let mut out = Vec::with_capacity(bw * bh);
    for by_i in 0..bh {
        for bx_i in 0..bw {
            let bx = bx_i * 8;
            let by = by_i * 8;
            let mut data = [0i16; 64];
            for y in 0..8 {
                let iy = (by + y).min(h.saturating_sub(1));
                for x in 0..8 {
                    let ix = (bx + x).min(w.saturating_sub(1));
                    data[y * 8 + x] = image[iy * w + ix] as i16 - 128;
                }
            }
            let mut blk = Block { data };
            blk.data = dct_reference(&blk);
            out.push(blk);
        }
    }
    out
}

/// 确定性整数量化（四舍五入、正负对称，与 c/quant.c 标量路径同构但不依赖 FFI）。
#[inline]
fn quantize_with_table(coef: &[i16; 64], table: &[i16; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    for i in 0..64 {
        let c = coef[i] as i32;
        let t = table[i] as i32;
        let v = if c >= 0 {
            (c + t / 2) / t
        } else {
            -((-c + t / 2) / t)
        };
        out[i] = v as i16;
    }
    out
}

/// 块活跃度：AC 系数绝对值之和（确定性统计量）。
fn block_activity(blk: &Block) -> u64 {
    let mut a = 0u64;
    for i in 1..64 {
        a += blk.data[ZIGZAG[i]].unsigned_abs() as u64;
    }
    a
}

// ---------------------------------------------------------------------------
// 逐块自适应表（g=1 的"原编码"表族）
// ---------------------------------------------------------------------------

/// 由块活跃度确定性构造该块自己的量化表。
/// DC 保持基准表；AC 按活跃度整体缩放（活跃度越高量化越粗），夹在 [1,255]。
fn per_block_table(base: &[i16; 64], activity: u64) -> [i16; 64] {
    // 活跃度 0      -> 800 permil
    // 活跃度 22000  -> 1350 permil（饱和）
    let raw = 800u32 + (activity as u32 / 40);
    let scale = raw.clamp(650, 1350);
    let mut t = [0i16; 64];
    t[0] = base[0];
    for i in 1..64 {
        let v = (base[i] as i32 * scale as i32 + 500) / 1000;
        t[i] = v.clamp(1, 255) as i16;
    }
    t
}

fn build_per_block_tables(blocks: &[Block], quality: u8) -> Vec<[i16; 64]> {
    let base = quant_table_for_quality(quality);
    blocks
        .iter()
        .map(|b| per_block_table(&base, block_activity(b)))
        .collect()
}

// ---------------------------------------------------------------------------
// 分类：2×2 tile 平滑/纹理，粗粒度按多数票聚合
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class { Smooth = 0, Texture = 1 }

/// 每个块的类别。g=1 不分类（返回全 Smooth 占位，实际不使用）。
fn classify_blocks(blocks: &[Block], w: usize, h: usize, g: u8, bias_permil: u32) -> Vec<Class> {
    let bw = blocks_wide(w);
    let bh = blocks_high(h);
    let n = blocks.len();
    let acts: Vec<u64> = blocks.iter().map(block_activity).collect();
    let total: u64 = acts.iter().sum();
    let mean = if n > 0 { total / n as u64 } else { 0 };
    let threshold = mean.saturating_mul(bias_permil as u64) / 1000;

    // 1) 2×2 tile 类别（最细一级共享）。
    let tile_bw = (bw + 1) / 2;
    let tile_bh = (bh + 1) / 2;
    let mut tile_class = vec![Class::Smooth; tile_bw * tile_bh];
    for ty in 0..tile_bh {
        for tx in 0..tile_bw {
            let mut sum = 0u64;
            let mut cnt = 0u64;
            for dy in 0..2 {
                for dx in 0..2 {
                    let bx = tx * 2 + dx;
                    let by = ty * 2 + dy;
                    if bx < bw && by < bh {
                        sum += acts[by * bw + bx];
                        cnt += 1;
                    }
                }
            }
            let tile_mean = if cnt > 0 { sum / cnt } else { 0 };
            tile_class[ty * tile_bw + tx] =
                if tile_mean > threshold { Class::Texture } else { Class::Smooth };
        }
    }

    // 2) g=2：块直接继承所属 2×2 tile。
    //    g=4/8：在更大 super-tile 内对 tile 类别投多数票（平票归 Smooth）。
    let mut out = vec![Class::Smooth; n];
    let group = if g <= 2 { 2usize } else { g as usize };
    let group_tiles = (group + 1) / 2;
    let super_bw = (bw + group - 1) / group;
    let super_bh = (bh + group - 1) / group;
    for sy in 0..super_bh {
        for sx in 0..super_bw {
            let mut smooth_votes = 0u32;
            let mut texture_votes = 0u32;
            for ty in sy * group_tiles..(sy + 1) * group_tiles {
                for tx in sx * group_tiles..(sx + 1) * group_tiles {
                    if ty < tile_bh && tx < tile_bw {
                        match tile_class[ty * tile_bw + tx] {
                            Class::Smooth => smooth_votes += 1,
                            Class::Texture => texture_votes += 1,
                        }
                    }
                }
            }
            let vote = if texture_votes > smooth_votes { Class::Texture } else { Class::Smooth };
            for by in sy * group..(sy * group + group).min(bh) {
                for bx in sx * group..(sx * group + group).min(bw) {
                    out[by * bw + bx] = vote;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 两套共享表：对同类逐块表取均值，再乘可配频率权重
// ---------------------------------------------------------------------------

fn average_tables(tables: &[[i16; 64]], indices: &[usize]) -> [i16; 64] {
    let mut acc = [0i64; 64];
    for &idx in indices {
        for i in 0..64 {
            acc[i] += tables[idx][i] as i64;
        }
    }
    let mut out = [0i16; 64];
    let n = indices.len().max(1) as i64;
    for i in 0..64 {
        out[i] = ((acc[i] + n / 2) / n).clamp(1, 255) as i16;
    }
    out
}

fn apply_profile(avg: &[i16; 64], profile: &[u8; 64]) -> [i16; 64] {
    let mut out = [0i16; 64];
    out[0] = avg[0]; // DC 不做频率加权
    for i in 1..64 {
        let v = (avg[i] as i32 * profile[i] as i32 + 50) / 100;
        out[i] = v.clamp(1, 255) as i16;
    }
    out
}

/// 构造 (平滑表, 纹理表)。某一类为空时退化为全图均值，保证表始终可用。
fn build_shared_tables(
    per_block: &[[i16; 64]],
    classes: &[Class],
    cfg: &SharingConfig,
) -> ([i16; 64], [i16; 64]) {
    let smooth_idx: Vec<usize> = classes
        .iter()
        .enumerate()
        .filter_map(|(i, c)| (*c == Class::Smooth).then_some(i))
        .collect();
    let texture_idx: Vec<usize> = classes
        .iter()
        .enumerate()
        .filter_map(|(i, c)| (*c == Class::Texture).then_some(i))
        .collect();
    let all: Vec<usize> = (0..per_block.len()).collect();

    let smooth_avg = if smooth_idx.is_empty() {
        average_tables(per_block, &all)
    } else {
        average_tables(per_block, &smooth_idx)
    };
    let texture_avg = if texture_idx.is_empty() {
        average_tables(per_block, &all)
    } else {
        average_tables(per_block, &texture_idx)
    };

    let smooth = apply_profile(&smooth_avg, &cfg.smooth_profile);
    let texture = apply_profile(&texture_avg, &cfg.texture_profile);
    (smooth, texture)
}

// ---------------------------------------------------------------------------
// RLE 载荷（与 .bg v1 RLE 同构：DC 2B + (run,level) 3B + EOB FF 00 00）
// ---------------------------------------------------------------------------

const EOB_RUN: u8 = 0xFF;
const MAX_RUN: u8 = 0xFE;

fn rle_encode(blocks: &[Block]) -> Vec<u8> {
    let mut out = Vec::with_capacity(blocks.len() * 8);
    for blk in blocks {
        let dc = blk.data[ZIGZAG[0]].to_le_bytes();
        out.extend_from_slice(&dc);
        let mut zero_run: u8 = 0;
        for i in 1..64 {
            let level = blk.data[ZIGZAG[i]];
            if level == 0 {
                if zero_run == MAX_RUN {
                    out.push(MAX_RUN);
                    out.extend_from_slice(&0i16.to_le_bytes());
                    zero_run = 0;
                }
                zero_run += 1;
            } else {
                out.push(zero_run);
                out.extend_from_slice(&level.to_le_bytes());
                zero_run = 0;
            }
        }
        out.push(EOB_RUN);
        out.extend_from_slice(&0i16.to_le_bytes());
    }
    out
}

fn rle_decode(buf: &[u8], n_blocks: usize) -> Option<Vec<Block>> {
    let mut pos = 0usize;
    let mut blocks = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        let mut blk = Block::new();
        if pos + 2 > buf.len() { return None; }
        blk.data[ZIGZAG[0]] = i16::from_le_bytes([buf[pos], buf[pos + 1]]);
        pos += 2;
        let mut ac = 1usize;
        loop {
            if pos + 3 > buf.len() { return None; }
            let run = buf[pos];
            let level = i16::from_le_bytes([buf[pos + 1], buf[pos + 2]]);
            pos += 3;
            if run == EOB_RUN && level == 0 { break; }
            for _ in 0..run {
                if ac < 64 { blk.data[ZIGZAG[ac]] = 0; ac += 1; }
            }
            if ac < 64 { blk.data[ZIGZAG[ac]] = level; ac += 1; }
        }
        blocks.push(blk);
    }
    Some(blocks)
}

// ---------------------------------------------------------------------------
// 实验容器 BQT（仅用于本研究，与 .bg 线性格式互不影响）
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 3] = b"BQT";
const CONTAINER_VERSION: u8 = 1;
const HEADER_LEN: usize = 3 + 1 + 2 + 2 + 1 + 1 + 2 + 2; // magic..table_count = 14

fn push_u16(out: &mut Vec<u8>, v: u16) { out.extend_from_slice(&v.to_le_bytes()); }
fn push_u32(out: &mut Vec<u8>, v: u32) { out.extend_from_slice(&v.to_le_bytes()); }

/// 编码结果。g=1 时 tables 为逐块表、indices 为恒等映射；
/// g≥2 时 tables 恰为 [平滑, 纹理] 两张、indices 为每块 1 bit 位图。
#[derive(Clone, Debug)]
struct SharedEncoding {
    bytes: Vec<u8>,
    tables: Vec<[i16; 64]>,
    indices: Vec<u16>,
    payload_len: usize,
    table_section_len: usize,
    index_bytes: usize,
    classes: Vec<Class>,
}

#[allow(clippy::too_many_arguments)]
fn serialize(
    w: usize,
    h: usize,
    quality: u8,
    g: u8,
    tables: &[[i16; 64]],
    indices: &[u16],
    payload: &[u8],
    classes: &[Class],
) -> SharedEncoding {
    let n = indices.len();
    assert!(tables.len() <= u16::MAX as usize);
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.push(CONTAINER_VERSION);
    push_u16(&mut out, w as u16);
    push_u16(&mut out, h as u16);
    out.push(quality);
    out.push(g);
    push_u16(&mut out, n as u16);
    push_u16(&mut out, tables.len() as u16);
    assert_eq!(out.len(), HEADER_LEN);

    let table_start = out.len();
    for t in tables {
        for v in t {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    let table_section_len = out.len() - table_start;

    let index_start = out.len();
    if tables.len() == 2 {
        out.push(1); // packed 1-bit
        let packed_len = (n + 7) / 8;
        let mut packed = vec![0u8; packed_len];
        for (i, &idx) in indices.iter().enumerate() {
            if idx == 1 {
                packed[i / 8] |= 1 << (i % 8);
            }
        }
        out.extend_from_slice(&packed);
    } else {
        out.push(0); // u16 per block
        for &idx in indices {
            push_u16(&mut out, idx);
        }
    }
    let index_bytes = out.len() - index_start;

    push_u32(&mut out, payload.len() as u32);
    out.extend_from_slice(payload);

    SharedEncoding {
        bytes: out,
        tables: tables.to_vec(),
        indices: indices.to_vec(),
        payload_len: payload.len(),
        table_section_len,
        index_bytes,
        classes: classes.to_vec(),
    }
}

/// 通用编码路径。g=1 时走逐块表（与 [`encode_original`] 逐字节一致）。
fn encode(
    image: &[u8],
    w: usize,
    h: usize,
    quality: u8,
    cfg: &SharingConfig,
) -> SharedEncoding {
    let g = cfg.granularity;
    let dct_blocks = forward_blocks(image, w, h);
    let per_block = build_per_block_tables(&dct_blocks, quality);

    if g == 1 {
        let mut quantized = Vec::with_capacity(dct_blocks.len());
        for (b, t) in dct_blocks.iter().zip(per_block.iter()) {
            quantized.push(Block { data: quantize_with_table(&b.data, t) });
        }
        let payload = rle_encode(&quantized);
        let indices: Vec<u16> = (0..per_block.len() as u16).collect();
        let classes = vec![Class::Smooth; per_block.len()];
        return serialize(w, h, quality, 1, &per_block, &indices, &payload, &classes);
    }

    let classes = classify_blocks(&dct_blocks, w, h, g, cfg.activity_bias);
    let (smooth, texture) = build_shared_tables(&per_block, &classes, cfg);
    let tables = vec![smooth, texture];

    let mut quantized = Vec::with_capacity(dct_blocks.len());
    let mut indices = Vec::with_capacity(dct_blocks.len());
    for (b, c) in dct_blocks.iter().zip(classes.iter()) {
        let idx = if *c == Class::Texture { 1u16 } else { 0u16 };
        indices.push(idx);
        quantized.push(Block { data: quantize_with_table(&b.data, &tables[idx as usize]) });
    }
    let payload = rle_encode(&quantized);
    serialize(w, h, quality, g, &tables, &indices, &payload, &classes)
}

/// 原编码参考实现：独立、直白的逐块表路径，不经过 tile/分类抽象。
/// g=1 的通用路径必须与本函数输出**逐字节相同**。
fn encode_original(image: &[u8], w: usize, h: usize, quality: u8) -> SharedEncoding {
    let base = quant_table_for_quality(quality);
    let bw = blocks_wide(w);
    let bh = blocks_high(h);
    let mut tables = Vec::with_capacity(bw * bh);
    let mut quantized = Vec::with_capacity(bw * bh);

    for by_i in 0..bh {
        for bx_i in 0..bw {
            let bx = bx_i * 8;
            let by = by_i * 8;
            let mut data = [0i16; 64];
            for y in 0..8 {
                let iy = (by + y).min(h.saturating_sub(1));
                for x in 0..8 {
                    let ix = (bx + x).min(w.saturating_sub(1));
                    data[y * 8 + x] = image[iy * w + ix] as i16 - 128;
                }
            }
            let mut blk = Block { data };
            blk.data = dct_reference(&blk);
            let table = per_block_table(&base, block_activity(&blk));
            let q = quantize_with_table(&blk.data, &table);
            tables.push(table);
            quantized.push(Block { data: q });
        }
    }
    let payload = rle_encode(&quantized);
    let indices: Vec<u16> = (0..tables.len() as u16).collect();
    let classes = vec![Class::Smooth; tables.len()];
    serialize(w, h, quality, 1, &tables, &indices, &payload, &classes)
}

#[derive(Clone, Debug)]
struct ContainerDecoded {
    w: usize,
    h: usize,
    quality: u8,
    g: u8,
    tables: Vec<[i16; 64]>,
    indices: Vec<u16>,
    pixels: Vec<u8>,
}

/// 容器解码：返回尺寸、表、索引、重建像素。
fn decode_container(buf: &[u8]) -> Option<ContainerDecoded> {
    if buf.len() < HEADER_LEN || &buf[0..3] != MAGIC {
        return None;
    }
    let mut pos = 3usize;
    let version = buf[pos];
    pos += 1;
    if version != CONTAINER_VERSION { return None; }
    let rd_u16 = |p: usize| -> Option<u16> {
        if p + 2 > buf.len() { return None; }
        Some(u16::from_le_bytes([buf[p], buf[p + 1]]))
    };
    let w = rd_u16(pos)? as usize; pos += 2;
    let h = rd_u16(pos)? as usize; pos += 2;
    let quality = buf[pos]; pos += 1;
    let g = buf[pos]; pos += 1;
    let n = rd_u16(pos)? as usize; pos += 2;
    let table_count = rd_u16(pos)? as usize; pos += 2;

    if pos + table_count * 128 > buf.len() { return None; }
    let mut tables = Vec::with_capacity(table_count);
    for _ in 0..table_count {
        let mut t = [0i16; 64];
        for v in t.iter_mut() {
            *v = i16::from_le_bytes([buf[pos], buf[pos + 1]]);
            pos += 2;
        }
        tables.push(t);
    }

    let index_kind = buf[pos]; pos += 1;
    let mut indices = Vec::with_capacity(n);
    if index_kind == 1 && table_count == 2 {
        let packed_len = (n + 7) / 8;
        if pos + packed_len > buf.len() { return None; }
        for i in 0..n {
            indices.push(if buf[pos + i / 8] & (1 << (i % 8)) != 0 { 1 } else { 0 });
        }
        pos += packed_len;
    } else {
        for _ in 0..n {
            let idx = rd_u16(pos)?; pos += 2;
            indices.push(idx);
        }
    }

    if pos + 4 > buf.len() { return None; }
    let payload_len = u32::from_le_bytes([buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]]) as usize;
    pos += 4;
    if pos + payload_len > buf.len() { return None; }
    let payload = &buf[pos..pos + payload_len];

    let mut qblocks = rle_decode(payload, n)?;
    let bw = blocks_wide(w);
    let mut pixels = vec![0u8; w * h];
    for (i, blk) in qblocks.iter_mut().enumerate() {
        let t = &tables[indices[i] as usize];
        for j in 0..64 {
            let v = blk.data[j] as i32 * t[j] as i32;
            blk.data[j] = v.clamp(-32768, 32767) as i16;
        }
        blk.data = idct_reference(&blk.data);
        let bx = (i % bw) * 8;
        let by = (i / bw) * 8;
        for y in 0..8 {
            for x in 0..8 {
                let px = bx + x;
                let py = by + y;
                if px < w && py < h {
                    let v = blk.data[y * 8 + x] + 128;
                    pixels[py * w + px] = v.clamp(0, 255) as u8;
                }
            }
        }
    }
    Some(ContainerDecoded { w, h, quality, g, tables, indices, pixels })
}

// ---------------------------------------------------------------------------
// 失真 / PSNR
// ---------------------------------------------------------------------------

fn sse(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as i32 - *y as i32;
            (d * d) as u64
        })
        .sum()
}

fn psnr_db(original: &[u8], reconstructed: &[u8]) -> f64 {
    let n = original.len() as f64;
    let mse = sse(original, reconstructed) as f64 / n;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

// ---------------------------------------------------------------------------
// 评测报告
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct GranularityReport {
    granularity: u8,
    total_bytes: usize,
    payload_bytes: usize,
    table_section_bytes: usize,
    index_bytes: usize,
    table_overhead_pct: f64,
    psnr_db: f64,
    sse: u64,
    rd_cost: f64,
    n_smooth: usize,
    n_texture: usize,
    n_tables: usize,
}

#[derive(Clone, Debug)]
struct SampleReport {
    name: String,
    width: usize,
    height: usize,
    quality: u8,
    original_bytes: usize,
    original_psnr_db: f64,
    per_granularity: Vec<GranularityReport>,
    best_granularity: u8,
}

/// 对一张图评估全部候选粒度，并按 λ 做率失真选择。
fn evaluate(
    name: &str,
    image: &[u8],
    w: usize,
    h: usize,
    quality: u8,
    cfg_base: &SharingConfig,
    candidates: &[u8],
) -> SampleReport {
    let original = encode_original(image, w, h, quality);
    let orig_decoded = decode_container(&original.bytes).expect("original container must decode");
    let original_psnr = psnr_db(image, &orig_decoded.pixels);

    let mut per_granularity = Vec::new();
    let mut best_g = 1u8;
    let mut best_cost = f64::MAX;

    for &g in candidates {
        let cfg = cfg_base.clone().with_granularity(g);
        let enc = encode(image, w, h, quality, &cfg);
        let dec = decode_container(&enc.bytes).expect("container must decode");
        let dist = sse(image, &dec.pixels);
        let psnr = psnr_db(image, &dec.pixels);
        let overhead = enc.table_section_len as f64 * 100.0 / enc.bytes.len() as f64;
        let rd = dist as f64 + cfg.rd_lambda_permil as f64 / 1000.0 * enc.bytes.len() as f64;
        let n_texture = enc.classes.iter().filter(|c| **c == Class::Texture).count();
        let report = GranularityReport {
            granularity: g,
            total_bytes: enc.bytes.len(),
            payload_bytes: enc.payload_len,
            table_section_bytes: enc.table_section_len,
            index_bytes: enc.index_bytes,
            table_overhead_pct: overhead,
            psnr_db: psnr,
            sse: dist,
            rd_cost: rd,
            n_smooth: enc.classes.len() - n_texture,
            n_texture,
            n_tables: enc.tables.len(),
        };
        if rd < best_cost {
            best_cost = rd;
            best_g = g;
        }
        per_granularity.push(report);
    }

    SampleReport {
        name: name.to_string(),
        width: w,
        height: h,
        quality,
        original_bytes: original.bytes.len(),
        original_psnr_db: original_psnr,
        per_granularity,
        best_granularity: best_g,
    }
}

// ---------------------------------------------------------------------------
// 四类小型合成素材（确定性、无 RNG、无外部库）
// ---------------------------------------------------------------------------

struct Sample {
    name: String,
    w: usize,
    h: usize,
    pixels: Vec<u8>,
}

impl Sample {
    /// 1) 平滑梯度：双向线性梯度 + 极小幅确定性扰动。
    fn smooth_gradient(size: usize) -> Self {
        let mut pixels = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let base = (x * 90 / size + y * 140 / size) / 2 + 20;
                let wobble = ((x + y) % 5) as i32 - 2;
                pixels[y * size + x] = (base as i32 + wobble).clamp(0, 255) as u8;
            }
        }
        Self { name: "smooth_gradient".into(), w: size, h: size, pixels }
    }

    /// 2) 强纹理：2 像素棋盘格（最高频能量）。
    fn checker_texture(size: usize) -> Self {
        let mut pixels = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                pixels[y * size + x] = if ((x / 2) + (y / 2)) % 2 == 0 { 40 } else { 215 };
            }
        }
        Self { name: "checker_texture".into(), w: size, h: size, pixels }
    }

    /// 3) 混合：上半平滑梯度、下半棋盘纹理。
    fn mixed_halves(size: usize) -> Self {
        let mut pixels = vec![0u8; size * size];
        for y in 0..size {
            for x in 0..size {
                pixels[y * size + x] = if y < size / 2 {
                    let base = (x * 180 / size) + 30;
                    let wobble = (x % 4) as i32 - 1;
                    (base as i32 + wobble).clamp(0, 255) as u8
                } else if ((x / 2) + (y / 2)) % 2 == 0 {
                    45
                } else {
                    210
                };
            }
        }
        Self { name: "mixed_halves".into(), w: size, h: size, pixels }
    }

    /// 4) 类自然图：两个正弦分量 + 软圆斑（多频混合，确定性）。
    fn naturalish(size: usize) -> Self {
        let mut pixels = vec![0u8; size * size];
        let s = size as f64;
        for y in 0..size {
            for x in 0..size {
                let fx = x as f64;
                let fy = y as f64;
                let wave1 = 40.0 * ((fx * 2.0 + fy) * std::f64::consts::PI / s).sin();
                let wave2 = 25.0 * ((fx - fy) * 5.0 * std::f64::consts::PI / s).cos();
                let dx = fx - s * 0.35;
                let dy = fy - s * 0.6;
                let blob = 55.0 * (-(dx * dx + dy * dy) / (2.0 * (s / 4.0).powi(2))).exp();
                let v = 128.0 + wave1 + wave2 + blob;
                pixels[y * size + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        Self { name: "naturalish".into(), w: size, h: size, pixels }
    }

    fn four_set(size: usize) -> Vec<Sample> {
        vec![
            Sample::smooth_gradient(size),
            Sample::checker_texture(size),
            Sample::mixed_halves(size),
            Sample::naturalish(size),
        ]
    }
}

// ---------------------------------------------------------------------------
// 单元测试 + 一次性实测
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const CANDIDATES: [u8; 4] = [1, 2, 4, 8];

    #[test]
    fn granularity_1_is_byte_identical_to_original() {
        // 多素材 × 多质量 × 含非 8 整除尺寸：g=1 通用路径必须逐字节等于独立参考实现。
        let mut samples = Sample::four_set(32);
        samples.push(Sample::smooth_gradient(20));
        samples.push(Sample::mixed_halves(28));
        for s in &samples {
            for &q in &[25u8, 50, 85] {
                let cfg = SharingConfig::default().with_granularity(1);
                let via_g1 = encode(&s.pixels, s.w, s.h, q, &cfg);
                let reference = encode_original(&s.pixels, s.w, s.h, q);
                assert_eq!(
                    via_g1.bytes, reference.bytes,
                    "g=1 byte identity broken on {} q={}", s.name, q
                );
                assert_eq!(via_g1.payload_len, reference.payload_len);
                assert_eq!(via_g1.tables.len(), s.w.div_ceil(8) * s.h.div_ceil(8));
            }
        }
    }

    #[test]
    fn container_roundtrips_all_granularities() {
        for s in Sample::four_set(32) {
            for &g in &CANDIDATES {
                let cfg = SharingConfig::default().with_granularity(g);
                let enc = encode(&s.pixels, s.w, s.h, 70, &cfg);
                let dec = decode_container(&enc.bytes).expect("decode failed");
                assert_eq!(dec.w, s.w);
                assert_eq!(dec.h, s.h);
                assert_eq!(dec.g, g);
                assert_eq!(dec.quality, 70);
                assert_eq!(dec.indices, enc.indices);
                assert_eq!(dec.tables.len(), enc.tables.len());
                for (a, b) in dec.tables.iter().zip(enc.tables.iter()) {
                    assert_eq!(a, b);
                }
                let dec2 = decode_container(&enc.bytes).unwrap();
                assert_eq!(dec.pixels, dec2.pixels);
            }
        }
    }

    #[test]
    fn shared_granularities_store_exactly_two_tables() {
        for s in Sample::four_set(32) {
            for &g in &[2u8, 4, 8] {
                let cfg = SharingConfig::default().with_granularity(g);
                let enc = encode(&s.pixels, s.w, s.h, 60, &cfg);
                assert_eq!(enc.tables.len(), 2, "g={} must use smooth+texture pair", g);
                assert!(enc.indices.iter().all(|&i| i <= 1));
            }
        }
    }

    #[test]
    fn classification_is_deterministic_and_monotone() {
        let s = Sample::mixed_halves(32);
        let blocks = forward_blocks(&s.pixels, s.w, s.h);
        let a = classify_blocks(&blocks, s.w, s.h, 2, 1000);
        let b = classify_blocks(&blocks, s.w, s.h, 2, 1000);
        assert_eq!(a, b);
        // 更高阈值 -> 平滑块不会更少。
        let relaxed = classify_blocks(&blocks, s.w, s.h, 2, 1500);
        let smooth_a = a.iter().filter(|c| **c == Class::Smooth).count();
        let smooth_r = relaxed.iter().filter(|c| **c == Class::Smooth).count();
        assert!(smooth_r >= smooth_a);
        let g4 = classify_blocks(&blocks, s.w, s.h, 4, 1000);
        assert_eq!(g4.len(), blocks.len());
    }

    #[test]
    fn custom_profiles_change_table_content() {
        let s = Sample::checker_texture(32);
        let blocks = forward_blocks(&s.pixels, s.w, s.h);
        let per_block = build_per_block_tables(&blocks, 70);
        let classes = classify_blocks(&blocks, s.w, s.h, 2, 1000);

        let cfg_default = SharingConfig::default();
        let (sm_def, tx_def) = build_shared_tables(&per_block, &classes, &cfg_default);

        let mut aggressive = cfg_default.clone();
        aggressive.texture_profile = DEFAULT_SMOOTH_PROFILE; // 故意换成更激进的权重
        let (sm2, tx2) = build_shared_tables(&per_block, &classes, &aggressive);
        assert_eq!(sm_def, sm2); // 平滑 profile 未改 -> 平滑表不变
        assert_ne!(tx_def, tx2);

        for t in [&sm_def, &tx_def, &sm2, &tx2] {
            assert!(t.iter().all(|&v| (1..=255).contains(&v)));
        }
    }

    #[test]
    fn rd_lambda_changes_selection_monotonically() {
        let s = Sample::naturalish(32);
        let cheap = SharingConfig { rd_lambda_permil: 1, ..Default::default() };
        let pricey = SharingConfig { rd_lambda_permil: 200_000, ..Default::default() };
        let r_cheap = evaluate(&s.name, &s.pixels, s.w, s.h, 60, &cheap, &CANDIDATES);
        let r_pricey = evaluate(&s.name, &s.pixels, s.w, s.h, 60, &pricey, &CANDIDATES);
        // λ 极大时 R 主导，最优粒度不可能比 λ 极小时更细。
        assert!(r_pricey.best_granularity >= r_cheap.best_granularity);
    }

    #[test]
    fn malformed_container_is_rejected() {
        assert!(decode_container(&[]).is_none());
        assert!(decode_container(b"BQT").is_none());
        let mut bad = b"BQT".to_vec();
        bad.push(CONTAINER_VERSION);
        bad.extend_from_slice(&32u16.to_le_bytes());
        bad.extend_from_slice(&32u16.to_le_bytes());
        bad.push(50);
        bad.push(1);
        bad.extend_from_slice(&16u16.to_le_bytes());
        bad.extend_from_slice(&16u16.to_le_bytes()); // 声明 16 张表但无内容
        assert!(decode_container(&bad).is_none());
    }

    /// 一次性实测：四类素材 × 四档粒度，打印体积 / PSNR / 表开销 / 选择分布。
    /// `cargo test --release -- --nocapture` 可见完整表格。
    #[test]
    fn measured_four_materials_report() {
        let samples = Sample::four_set(32);
        let quality = 60u8;
        let cfg = SharingConfig::default();
        let mut all = Vec::new();
        for s in &samples {
            all.push(evaluate(&s.name, &s.pixels, s.w, s.h, quality, &cfg, &CANDIDATES));
        }

        println!();
        println!("================ BQT 共享量化表实测（32x32 灰度, quality={quality}） ================");
        for r in &all {
            assert_eq!(r.quality, quality);
            println!("\n[{}] {}x{}  quality={}  原编码(g=1) {} B, PSNR {:.2} dB",
                r.name, r.width, r.height, r.quality, r.original_bytes, r.original_psnr_db);
            println!("  g   总B   载荷B  表区B  索引B  表开销%  PSNR dB   ΔPSNR dB  SSE    平滑/纹理  表数  RD代价");
            for gr in &r.per_granularity {
                let delta = r.original_psnr_db - gr.psnr_db;
                let mark = if gr.granularity == r.best_granularity { "*" } else { " " };
                println!(
                    "  {g:<1}{mk} {tot:>5} {pay:>6} {tab:>6} {idx:>5} {ov:>7.1} {ps:>8.2} {dl:>9.3} {sse:>6} {sm:>4}/{tx:<4} {nt:>4} {rd:>10.1}",
                    g = gr.granularity,
                    mk = mark,
                    tot = gr.total_bytes,
                    pay = gr.payload_bytes,
                    tab = gr.table_section_bytes,
                    idx = gr.index_bytes,
                    ov = gr.table_overhead_pct,
                    ps = gr.psnr_db,
                    dl = delta,
                    sse = gr.sse,
                    sm = gr.n_smooth,
                    tx = gr.n_texture,
                    nt = gr.n_tables,
                    rd = gr.rd_cost,
                );
            }
            println!("  -> RD(λ={}/1000) 选择 g={}", cfg.rd_lambda_permil, r.best_granularity);
        }

        // 硬性不变量（任何一项失败即测试失败）。
        for r in &all {
            let g1 = &r.per_granularity[0];
            assert_eq!(g1.granularity, 1);
            assert_eq!(g1.total_bytes, r.original_bytes);
            assert!((g1.psnr_db - r.original_psnr_db).abs() < 1e-9);
            for gr in &r.per_granularity {
                if gr.granularity >= 2 {
                    // 共享后表区必须显著小于原编码，且总体积不大于原编码。
                    assert!(gr.table_section_bytes < g1.table_section_bytes);
                    assert!(gr.total_bytes <= g1.total_bytes);
                    // PSNR 损失有界（实验容器内，允许最大 3 dB）。
                    assert!(r.original_psnr_db - gr.psnr_db <= 3.0,
                        "{} g={} PSNR loss too large", r.name, gr.granularity);
                }
            }
        }
    }
}
