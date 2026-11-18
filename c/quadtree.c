/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * quadtree.c — self-contained quadtree partition codec (grayscale, 8 bpp).
 *
 * Motivation
 * ----------
 * The main codec partitions every image into a fixed grid of blocks, so large
 * flat areas (solid backgrounds, UI panels) pay for blocks that carry almost
 * no information.  This module adds an adaptive partition:
 *
 *   1. The image is covered by 64x64 root tiles (edge pixels replicated).
 *   2. Each tile is a quadtree.  A block is recursively subdivided into four
 *      equal children while ALL of the following hold:
 *        - block depth < configurable max depth (hard cap, always respected),
 *        - block side  > 4 (minimum leaf size),
 *        - block variance > variance threshold (flat blocks stop immediately),
 *        - the four children together give a lower rate-distortion cost than
 *          keeping the block as one leaf (bottom-up RD guard), so smooth
 *          gradients are not needlessly shattered into four leaves.
 *   3. Every leaf block is quantized independently: its own quantizer step,
 *      chosen from the leaf's local standard deviation (4 modes, 2 bits in
 *      the stream), applied to the block mean.
 *   4. The partition tree is serialized into the stream header section and
 *      the decoder reconstructs the exact same tree (no analysis needed).
 *
 * A fixed 16x16 partition encoder using the identical entropy coder and
 * quantizer logic is included so that size/PSNR comparisons isolate the
 * effect of the adaptive partition.
 *
 * No third-party code and no network access are used: only libc and libm.
 *
 * The demonstration harness (sample generation + acceptance metrics) is
 * compiled with -DBITGRAIN_QT_DEMO; without it this file is a plain library.
 */
#ifndef BITGRAIN_QT_DEMO
/* Library consumers get non-static entry points. */
#define QT_API
#else
#define QT_API
#endif

#if defined(BITGRAIN_QT_DEMO) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L /* clock_gettime */
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Public types and entry points                                      */
/* ------------------------------------------------------------------ */

#define QT_TILE_SIZE        64    /* root tile side                          */
#define QT_MIN_LEAF_SIZE     4    /* smallest quadtree leaf side             */
#define QT_MAX_DEPTH_CAP     4    /* depth 4 => smallest leaf 64/16 = 4 px   */
#define QT_VARIANCE_DFLT     4.0  /* default stddev threshold                */
#define QT_QUALITY_DFLT      85

typedef struct {
    uint32_t leaves_by_depth[QT_MAX_DEPTH_CAP + 1]; /* leaf counts per depth */
    uint32_t internal_nodes;
    uint32_t root_tiles;
    uint32_t min_possible_leaves;   /* == root_tiles (all-flat lower bound)  */
} qt_stats;

/*
 * Quadtree encode.
 * max_depth:   0..QT_MAX_DEPTH_CAP, subdivision stops at this depth.
 * quality:     1..100 (0 -> QT_QUALITY_DFLT); drives quantizer steps and the
 *              RD Lagrange multiplier.
 * var_thresh:  stddev threshold; blocks with stddev below it never split.
 *              <= 0 selects the default.
 * Returns stream length, or -1 on error.
 */
QT_API int qt_encode(const uint8_t *img, uint32_t w, uint32_t h,
                     uint8_t quality, uint8_t max_depth, double var_thresh,
                     uint8_t *out, size_t out_cap, qt_stats *st);

/* Decode a quadtree stream. Returns 0 on success, -1 on error. */
QT_API int qt_decode(const uint8_t *buf, size_t len,
                     uint8_t *out_img, uint32_t w, uint32_t h,
                     uint8_t *out_max_depth);

/* Fixed 16x16 grid reference encoder (same entropy backend). */
QT_API int fx16_encode(const uint8_t *img, uint32_t w, uint32_t h,
                       uint8_t quality, uint8_t *out, size_t out_cap,
                       uint32_t *out_blocks);

QT_API int fx16_decode(const uint8_t *buf, size_t len,
                       uint8_t *out_img, uint32_t w, uint32_t h);

/* ------------------------------------------------------------------ */
/* Bit writer / reader (MSB-first, JPEG-style)                        */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *buf;
    size_t   cap_bytes;
    size_t   pos;        /* byte position              */
    uint64_t acc;        /* pending bits, MSB aligned  */
    int      nbits;      /* valid bits in acc          */
    int      overflow;
} bitw;

static void bw_init(bitw *bw, uint8_t *buf, size_t cap)
{
    bw->buf = buf; bw->cap_bytes = cap; bw->pos = 0;
    bw->acc = 0; bw->nbits = 0; bw->overflow = 0;
}

static uint64_t mask64(unsigned n)
{
    return n >= 64 ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << n) - 1ULL);
}
static uint32_t mask32(unsigned n)
{
    return n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u);
}

static void bw_put(bitw *bw, uint64_t value, unsigned n)
{
    if (n == 0) return;
    if (bw->overflow) return;
    if (bw->pos + (size_t)((bw->nbits + n + 7) >> 3) > bw->cap_bytes) {
        bw->overflow = 1; return;
    }
    bw->acc = (bw->acc << n) | (value & mask64(n));
    bw->nbits += (int)n;
    while (bw->nbits >= 8) {
        bw->nbits -= 8;
        bw->buf[bw->pos++] = (uint8_t)(bw->acc >> bw->nbits);
    }
}

static size_t bw_finish(bitw *bw)
{
    if (bw->overflow) return (size_t)-1;
    if (bw->nbits > 0) {
        if (bw->pos >= bw->cap_bytes) return (size_t)-1;
        bw->buf[bw->pos++] = (uint8_t)((bw->acc << (8 - bw->nbits)) |
                                       ((1u << (8 - bw->nbits)) - 1u));
    }
    return bw->pos;
}

typedef struct {
    const uint8_t *buf;
    size_t len;
    size_t pos;
    uint32_t acc;
    int nbits;
    int error;
} bitr;

static void br_init(bitr *br, const uint8_t *buf, size_t len)
{
    br->buf = buf; br->len = len; br->pos = 0;
    br->acc = 0; br->nbits = 0; br->error = 0;
}

static uint32_t br_get(bitr *br, unsigned n)
{
    if (n == 0) return 0;
    while ((unsigned)br->nbits < n) {
        if (br->pos >= br->len) { br->error = 1; return 0; }
        br->acc = (br->acc << 8) | br->buf[br->pos++];
        br->nbits += 8;
    }
    br->nbits -= (int)n;
    return (br->acc >> br->nbits) & mask32(n);
}

/* ------------------------------------------------------------------ */
/* Adaptive Golomb-Rice coding (LOCO-I style k adaptation)            */
/* ------------------------------------------------------------------ */

#define RICE_PERIOD 16
#define RICE_K_MAX  15
#define RICE_K_EST  2   /* fixed k used by the RD bit-cost estimator */

typedef struct { uint32_t sum; unsigned k, syms; } rice;

static void rice_init(rice *r) { r->sum = 0; r->k = 1; r->syms = 0; }

static uint32_t rice_map_signed(int32_t v)
{
    /* 0, -1, 1, -2, 2 ... -> 0, 1, 2, 3, 4 ... (unsigned math, no signed
     * left shift which would be undefined behaviour for negative inputs). */
    if (v <= 0)
        return (uint32_t)(-(int64_t)v) << 1;
    return ((uint32_t)v << 1) - 1u;
}

static int32_t rice_unmap_signed(uint32_t u)
{
    return (u & 1u) ? (int32_t)((u + 1u) >> 1) : -(int32_t)(u >> 1);
}

static void rice_put(bitw *bw, rice *r, uint32_t x)
{
    unsigned k = r->k;
    uint32_t q = x >> k;
    /* unary part: q zero bits then a one */
    while (q >= 32) { bw_put(bw, 0, 32); q -= 32; }
    bw_put(bw, 1u, (unsigned)q + 1u);
    if (k) bw_put(bw, x & mask32(k), k);

    r->sum += x;
    if (++r->syms == RICE_PERIOD) {
        r->syms = 0;
        if (r->k > 0 && r->sum < ((uint32_t)RICE_PERIOD << (r->k - 1)))
            r->k--;
        else if (r->sum > ((uint32_t)RICE_PERIOD << r->k) && r->k < RICE_K_MAX)
            r->k++;
        r->sum = 0;
    }
}

static uint32_t rice_get(bitr *br, rice *r)
{
    unsigned k = r->k;
    uint32_t q = 0;
    while (br_get(br, 1) == 0 && !br->error) {
        if (++q == 0xFFFFFFFFu) { br->error = 1; break; }
    }
    uint32_t x = (q << k) | (k ? br_get(br, k) : 0);

    r->sum += x;
    if (++r->syms == RICE_PERIOD) {
        r->syms = 0;
        if (r->k > 0 && r->sum < ((uint32_t)RICE_PERIOD << (r->k - 1)))
            r->k--;
        else if (r->sum > ((uint32_t)RICE_PERIOD << r->k) && r->k < RICE_K_MAX)
            r->k++;
        r->sum = 0;
    }
    return x;
}

/* Rate estimator for the RD pass: context-free Rice at k = RICE_K_EST. */
static unsigned est_rice_bits(uint32_t x)
{
    return (x >> RICE_K_EST) + 1u + RICE_K_EST;
}

/* ------------------------------------------------------------------ */
/* Quantization                                                        */
/* ------------------------------------------------------------------ */

/* Base steps for the four per-leaf quantizer modes. */
static const double qstep_base[4] = { 1.0, 2.0, 4.0, 8.0 };

static double quality_factor(uint8_t quality)
{
    int q = quality ? quality : QT_QUALITY_DFLT;
    double f = (100.0 - q) * 0.12 + 0.5;
    if (f < 0.2) f = 0.2;
    if (f > 12.0) f = 12.0;
    return f;
}

/* Mode chosen from a leaf block's local standard deviation. */
static int quant_mode_from_sd(double sd, double qf)
{
    double t0 = 1.5 * qf, t1 = 5.0 * qf, t2 = 15.0 * qf;
    if (sd < t0) return 0;
    if (sd < t1) return 1;
    if (sd < t2) return 2;
    return 3;
}

static int32_t round_div(double v, double step)
{
    return (int32_t)lround(v / step);
}

static uint8_t reconstruct_level(int32_t level, double step)
{
    double v = level * step;
    long r = lround(v);
    if (r < 0) r = 0;
    if (r > 255) r = 255;
    return (uint8_t)r;
}

/* ------------------------------------------------------------------ */
/* Block statistics (edge pixels replicated, SSE over real pixels)    */
/* ------------------------------------------------------------------ */

typedef struct {
    double mean, var, sd;
    uint64_t sse_const;   /* SSE of the best constant approximation */
    int      mode;
    int32_t  qmean;
    uint8_t  recon;       /* reconstructed constant value           */
} block_info;

static block_info analyze_block(const uint8_t *img, uint32_t w, uint32_t h,
                                uint32_t x0, uint32_t y0, uint32_t size,
                                double qf)
{
    uint64_t sum = 0, sum2 = 0, npix = 0;
    uint32_t xe = x0 + size, ye = y0 + size;
    if (xe > w) xe = w;
    if (ye > h) ye = h;
    for (uint32_t y = y0; y < ye; y++) {
        const uint8_t *row = img + (size_t)y * w;
        for (uint32_t x = x0; x < xe; x++) {
            uint32_t v = row[x];
            sum += v;
            sum2 += (uint64_t)v * v;
            npix++;
        }
    }
    block_info bi;
    bi.mean = (double)sum / (double)npix;
    double mean2 = (double)sum2 / (double)npix;
    double var = mean2 - bi.mean * bi.mean;
    if (var < 0.0) var = 0.0;
    bi.var = var;
    bi.sd = sqrt(var);
    bi.mode = quant_mode_from_sd(bi.sd, qf);
    double step = qstep_base[bi.mode] * qf;
    bi.qmean = round_div(bi.mean, step);
    bi.recon = reconstruct_level(bi.qmean, step);

    uint64_t sse = 0;
    for (uint32_t y = y0; y < ye; y++) {
        const uint8_t *row = img + (size_t)y * w;
        for (uint32_t x = x0; x < xe; x++) {
            int d = (int)row[x] - (int)bi.recon;
            sse += (uint64_t)(d * d);
        }
    }
    bi.sse_const = sse;
    return bi;
}

/* Spatial predictor (MED/JPEG-LS style) on the reconstructed image. */
static int32_t predict_level(const uint8_t *rec, uint32_t w, uint32_t h,
                             uint32_t x0, uint32_t y0, double step)
{
    /*
     * Edge blocks may start on or beyond the image boundary. Clamp access
     * coordinates to the last real pixel (edge replication); the flag logic
     * then treats an off-image row/column as a missing neighbour.
     */
    uint32_t cx = x0 < w ? x0 : w - 1;
    uint32_t cy = y0 < h ? y0 : h - 1;
    long a = 0, b = 0, c = 0;
    int have_a = 0, have_b = 0, have_c = 0;
    if (x0 > 0 && x0 < w)     { a = rec[(size_t)cy * w + (cx - 1)]; have_a = 1; }
    if (y0 > 0 && y0 < h)     { b = rec[(size_t)(cy - 1) * w + cx]; have_b = 1; }
    if (x0 > 0 && y0 > 0 &&
        x0 < w && y0 < h)     { c = rec[(size_t)(cy - 1) * w + (cx - 1)]; have_c = 1; }

    long pred;
    if (!have_a && !have_b) pred = 128;
    else if (!have_a)       pred = b;
    else if (!have_b)       pred = a;
    else {
        long mx = a > b ? a : b, mn = a < b ? a : b;
        if (have_c && c >= mx)      pred = mn;
        else if (have_c && c <= mn) pred = mx;
        else                        pred = a + b - (have_c ? c : 0);
    }
    if (pred < 0) pred = 0;
    if (pred > 255) pred = 255;
    return round_div((double)pred, step);
}

/* ------------------------------------------------------------------ */
/* Quadtree analysis arena                                             */
/* ------------------------------------------------------------------ */

typedef struct qt_node {
    uint32_t x0, y0, size;
    uint8_t  depth;
    uint8_t  is_leaf;
    uint8_t  flat_leaf;  /* stopped on the flatness rule: mode 0 implicit */
    uint8_t  mode;
    int32_t  qmean;
    uint8_t  recon;
    uint32_t child[4];
    double   rd_cost;
} qt_node;

typedef struct {
    qt_node  *nodes;
    size_t    cap, count;
    const uint8_t *img;
    uint32_t  w, h;
    double    qf, lambda, var_thresh;
    uint8_t   max_depth;
} qt_builder;

static uint32_t alloc_node(qt_builder *b)
{
    if (b->count >= b->cap) return UINT32_MAX;
    uint32_t id = (uint32_t)b->count++;
    memset(&b->nodes[id], 0, sizeof(qt_node));
    return id;
}

static double leaf_rd_cost(qt_builder *b, const block_info *bi,
                           uint32_t x0, uint32_t y0, int mode_implicit)
{
    double step = qstep_base[bi->mode] * b->qf;
    /*
     * Estimate the MED predictor from source pixels: the encoder emits the
     * residual against reconstructed neighbours, which track the source
     * closely at the configured quality. This keeps the RD estimator in the
     * same residual regime as the real Rice coder.
     */
    int32_t pred = predict_level(b->img, b->w, b->h, x0, y0, step);
    uint32_t residual = rice_map_signed(bi->qmean - pred);
    /* Flat leaves make their mode 0 implicit and carry no mode bits. */
    double mode_bits = mode_implicit ? 0.0 : 2.0;
    double bits = 1.0 /* leaf flag */ + mode_bits +
                  (double)est_rice_bits(residual);
    return (double)bi->sse_const + b->lambda * bits;
}

static uint32_t build_node(qt_builder *b, uint32_t x0, uint32_t y0,
                           uint32_t size, uint8_t depth)
{
    uint32_t id = alloc_node(b);
    if (id == UINT32_MAX) return UINT32_MAX;
    qt_node *nd = &b->nodes[id];
    nd->x0 = x0; nd->y0 = y0; nd->size = size; nd->depth = depth;

    block_info bi = analyze_block(b->img, b->w, b->h, x0, y0, size, b->qf);
    nd->mode = (uint8_t)bi.mode;
    nd->qmean = bi.qmean;
    nd->recon = bi.recon;

    int can_split = (depth < b->max_depth) && (size > QT_MIN_LEAF_SIZE) &&
                    (size % 2 == 0);
    int above_variance = bi.var > b->var_thresh;

    if (!can_split || !above_variance) {
        nd->is_leaf = 1;
        /*
         * Only structural caps (max depth / min size / odd side) are known
         * to the decoder without pixel access, so only those leaves use
         * implicit mode 0. Variance-gated leaves still emit their mode.
         */
        nd->flat_leaf = !can_split;
        if (nd->flat_leaf) {
            nd->mode = 0;
            double step = qstep_base[0] * b->qf;
            nd->qmean = round_div(bi.mean, step);
            nd->recon = reconstruct_level(nd->qmean, step);
            bi.mode = 0; bi.qmean = nd->qmean; bi.recon = nd->recon;
            uint64_t sse = 0;
            uint32_t xe = x0 + size, ye = y0 + size;
            if (xe > b->w) xe = b->w;
            if (ye > b->h) ye = b->h;
            for (uint32_t y = y0; y < ye; y++) {
                const uint8_t *row = b->img + (size_t)y * b->w;
                for (uint32_t x = x0; x < xe; x++) {
                    int d = (int)row[x] - (int)nd->recon;
                    sse += (uint64_t)(d * d);
                }
            }
            bi.sse_const = sse;
        }
        nd->rd_cost = leaf_rd_cost(b, &bi, x0, y0, nd->flat_leaf);
        return id;
    }

    /* Recurse first (bottom-up), then apply the RD guard. */
    uint32_t half = size / 2;
    double children_cost = 1.0; /* internal-node flag bit */
    int failed = 0;
    for (int q = 0; q < 4; q++) {
        uint32_t cx = x0 + (q & 1) * half;
        uint32_t cy = y0 + (q >> 1) * half;
        nd->child[q] = build_node(b, cx, cy, half, (uint8_t)(depth + 1));
        if (nd->child[q] == UINT32_MAX) { failed = 1; break; }
        children_cost += b->nodes[nd->child[q]].rd_cost;
    }
    if (failed) return UINT32_MAX;

    double leaf_cost = leaf_rd_cost(b, &bi, x0, y0, 0);
    if (leaf_cost <= children_cost) {
        /* Keep as leaf: children were analyzed only for the RD decision. */
        nd->is_leaf = 1;
        nd->flat_leaf = 0;
        nd->rd_cost = leaf_cost;
    } else {
        nd->is_leaf = 0;
        nd->rd_cost = children_cost;
    }
    return id;
}

/* Stats are collected from the FINAL tree, so rejected speculative
 * subtrees never affect the reported partition. */
static void count_tree(const qt_node *nodes, const qt_node *nd, qt_stats *st)
{
    if (nd->is_leaf) {
        st->leaves_by_depth[nd->depth]++;
        return;
    }
    st->internal_nodes++;
    for (int q = 0; q < 4; q++)
        count_tree(nodes, &nodes[nd->child[q]], st);
}

/* ------------------------------------------------------------------ */
/* Stream headers                                                      */
/* ------------------------------------------------------------------ */

#define QT_MAGIC0 'Q', 'T'
#define FX_MAGIC0 'F', '1'
#define QT_VERSION 1

static void put_u16le(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
}
static void put_u32le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint32_t get_u32le(const uint8_t *p)
{ return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

/* ------------------------------------------------------------------ */
/* Quadtree emission / decoding                                        */
/* ------------------------------------------------------------------ */

static void emit_node(bitw *bw, rice *rc, const qt_node *nodes,
                      const qt_node *nd, const uint8_t *rec, uint32_t w,
                      uint32_t h, double qf)
{
    if (nd->is_leaf) {
        bw_put(bw, 0, 1);                       /* leaf flag */
        if (!nd->flat_leaf)
            bw_put(bw, nd->mode, 2);            /* per-leaf quantizer mode */
        double step = qstep_base[nd->mode] * qf;
        int32_t pred = predict_level(rec, w, h, nd->x0, nd->y0, step);
        rice_put(bw, rc, rice_map_signed(nd->qmean - pred));

        uint32_t xe = nd->x0 + nd->size, ye = nd->y0 + nd->size;
        if (xe > w) xe = w;
        if (ye > h) ye = h;
        for (uint32_t y = nd->y0; y < ye; y++)
            memset((void *)(rec + (size_t)y * w + nd->x0), nd->recon,
                   xe - nd->x0);
        return;
    }
    bw_put(bw, 1, 1);                           /* internal flag */
    for (int q = 0; q < 4; q++)
        emit_node(bw, rc, nodes, &nodes[nd->child[q]], rec, w, h, qf);
}

static int decode_node(bitr *br, rice *rc, uint8_t *rec, uint32_t w,
                       uint32_t h, uint32_t x0, uint32_t y0,
                       uint32_t size, uint8_t depth, uint8_t max_depth,
                       double qf, qt_stats *st)
{
    uint32_t flag = br_get(br, 1);
    if (br->error) return -1;
    if (flag == 0) {
        /*
         * Matches the encoder's can_split condition: when subdivision is
         * structurally impossible (depth cap or minimum leaf size) the leaf
         * is encoded with implicit mode 0; all other leaves carry 2 explicit
         * quantizer-mode bits.
         */
        int boundary = (depth >= max_depth) || (size <= QT_MIN_LEAF_SIZE) ||
                       (size % 2u != 0u);
        uint32_t m = boundary ? 0u : br_get(br, 2);
        if (m > 3) return -1;
        double step = qstep_base[m] * qf;
        int32_t pred = predict_level(rec, w, h, x0, y0, step);
        int32_t level = pred + rice_unmap_signed(rice_get(br, rc));
        uint8_t recon = reconstruct_level(level, step);

        if (st && depth <= QT_MAX_DEPTH_CAP) st->leaves_by_depth[depth]++;
        uint32_t xe = x0 + size, ye = y0 + size;
        if (xe > w) xe = w;
        if (ye > h) ye = h;
        for (uint32_t y = y0; y < ye; y++)
            memset(rec + (size_t)y * w + x0, recon, xe - x0);
        return 0;
    }
    if ((depth >= max_depth) || (size <= QT_MIN_LEAF_SIZE) ||
        (size % 2u != 0u))
        return -1;                          /* structure violates header caps */
    if (st) st->internal_nodes++;
    uint32_t half = size / 2;
    for (int q = 0; q < 4; q++) {
        uint32_t cx = x0 + (q & 1) * half;
        uint32_t cy = y0 + (q >> 1) * half;
        if (decode_node(br, rc, rec, w, h, cx, cy, half,
                        (uint8_t)(depth + 1), max_depth, qf, st))
            return -1;
    }
    return 0;
}

/*
 * RD Lagrange multiplier: lambda = lambda_scale * qf^2. Overridden by the
 * demo harness for tuning; the stream header does not depend on it.
 *
 * The scale is the rate-vs-distortion exchange rate per estimated bit,
 * expressed in squared-error units. 8000 is the knee where all four
 * reference materials (solid / screenshot / photo / gradient) compress
 * below the fixed 16x16 partition while retaining comparable PSNR.
 */
double qt_lambda_scale = 8000.0;

QT_API int qt_encode(const uint8_t *img, uint32_t w, uint32_t h,
                     uint8_t quality, uint8_t max_depth, double var_thresh,
                     uint8_t *out, size_t out_cap, qt_stats *st)
{
    if (!img || !out || w == 0 || h == 0 || max_depth > QT_MAX_DEPTH_CAP)
        return -1;
    if (quality > 100) return -1;

    double qf = quality_factor(quality);
    double lambda = qt_lambda_scale * qf * qf;
    double vt = (var_thresh > 0.0) ? var_thresh : QT_VARIANCE_DFLT;

    qt_stats local_st;
    memset(&local_st, 0, sizeof(local_st));
    qt_stats *stp = st ? st : &local_st;

    /* Worst case: every 4x4 cell becomes its own node plus the ancestors.
     * The +64 covers small/odd dimensions where the division undercounts. */
    size_t cells = ((size_t)w * h) /
                   ((size_t)QT_MIN_LEAF_SIZE * QT_MIN_LEAF_SIZE);
    size_t arena_cap = cells * 2 + 64;
    if (arena_cap < 64) arena_cap = 64;
    qt_node *nodes = (qt_node *)malloc(arena_cap * sizeof(qt_node));
    if (!nodes) return -1;

    qt_builder b;
    b.nodes = nodes; b.cap = arena_cap; b.count = 0;
    b.img = img; b.w = w; b.h = h; b.qf = qf; b.lambda = lambda;
    b.var_thresh = vt * vt; b.max_depth = max_depth;

    uint32_t tx = (w + QT_TILE_SIZE - 1) / QT_TILE_SIZE;
    uint32_t ty = (h + QT_TILE_SIZE - 1) / QT_TILE_SIZE;
    stp->root_tiles = tx * ty;
    stp->min_possible_leaves = tx * ty;

    uint32_t *roots = (uint32_t *)malloc((size_t)tx * ty * sizeof(uint32_t));
    uint8_t *rec = (uint8_t *)calloc((size_t)w * h, 1);
    if (!roots || !rec) { free(nodes); free(roots); free(rec); return -1; }

    int rc_fail = 0;
    for (uint32_t tyi = 0; tyi < ty; tyi++) {
        for (uint32_t txi = 0; txi < tx; txi++) {
            uint32_t id = build_node(&b, txi * QT_TILE_SIZE,
                                     tyi * QT_TILE_SIZE, QT_TILE_SIZE, 0);
            if (id == UINT32_MAX) { rc_fail = 1; break; }
            roots[tyi * tx + txi] = id;
        }
        if (rc_fail) break;
    }
    if (rc_fail) { free(nodes); free(roots); free(rec); return -1; }
    for (uint32_t i = 0; i < tx * ty; i++)
        count_tree(nodes, &nodes[roots[i]], stp);

    /* Header: 15 bytes. */
    if (out_cap < 15) { free(nodes); free(roots); free(rec); return -1; }
    out[0] = 'Q'; out[1] = 'T'; out[2] = QT_VERSION;
    put_u32le(out + 3, w);
    put_u32le(out + 7, h);
    out[11] = max_depth;
    out[12] = quality ? quality : QT_QUALITY_DFLT;
    put_u16le(out + 13, (uint16_t)lround(vt * 16.0));

    bitw bw;
    bw_init(&bw, out + 15, out_cap - 15);
    rice rc;
    rice_init(&rc);
    for (uint32_t i = 0; i < tx * ty; i++)
        emit_node(&bw, &rc, nodes, &nodes[roots[i]], rec, w, h, qf);
    size_t body = bw_finish(&bw);
    free(nodes); free(roots); free(rec);
    if (body == (size_t)-1) return -1;
    return (int)(15 + body);
}

QT_API int qt_decode(const uint8_t *buf, size_t len,
                     uint8_t *out_img, uint32_t w, uint32_t h,
                     uint8_t *out_max_depth)
{
    if (!buf || !out_img || len < 15 || buf[0] != 'Q' || buf[1] != 'T' ||
        buf[2] != QT_VERSION)
        return -1;
    uint32_t sw = get_u32le(buf + 3), sh = get_u32le(buf + 7);
    if (sw != w || sh != h) return -1;
    uint8_t max_depth = buf[11], quality = buf[12];
    if (max_depth > QT_MAX_DEPTH_CAP) return -1;
    double qf = quality_factor(quality);

    bitr br;
    br_init(&br, buf + 15, len - 15);
    rice rc;
    rice_init(&rc);

    uint32_t tx = (w + QT_TILE_SIZE - 1) / QT_TILE_SIZE;
    uint32_t ty = (h + QT_TILE_SIZE - 1) / QT_TILE_SIZE;
    for (uint32_t tyi = 0; tyi < ty; tyi++) {
        for (uint32_t txi = 0; txi < tx; txi++) {
            if (decode_node(&br, &rc, out_img, w, h,
                            txi * QT_TILE_SIZE, tyi * QT_TILE_SIZE,
                            QT_TILE_SIZE, 0, max_depth, qf, NULL))
                return -1;
        }
    }
    if (out_max_depth) *out_max_depth = max_depth;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Fixed 16x16 reference partition                                     */
/* ------------------------------------------------------------------ */

#define FX_BLOCK_SIZE 16
#define FX_HDR 12

QT_API int fx16_encode(const uint8_t *img, uint32_t w, uint32_t h,
                       uint8_t quality, uint8_t *out, size_t out_cap,
                       uint32_t *out_blocks)
{
    if (!img || !out || w == 0 || h == 0 || quality > 100) return -1;
    double qf = quality_factor(quality);
    if (out_cap < FX_HDR) return -1;

    out[0] = 'F'; out[1] = '1'; out[2] = QT_VERSION;
    put_u32le(out + 3, w);
    put_u32le(out + 7, h);
    out[11] = quality ? quality : QT_QUALITY_DFLT;

    bitw bw;
    bw_init(&bw, out + FX_HDR, out_cap - FX_HDR);
    rice rc;
    rice_init(&rc);

    uint8_t *rec = (uint8_t *)calloc((size_t)w * h, 1);
    if (!rec) return -1;

    uint32_t bx = (w + FX_BLOCK_SIZE - 1) / FX_BLOCK_SIZE;
    uint32_t by = (h + FX_BLOCK_SIZE - 1) / FX_BLOCK_SIZE;
    uint32_t nblocks = 0;
    int fail = 0;
    for (uint32_t byi = 0; byi < by && !fail; byi++) {
        for (uint32_t bxi = 0; bxi < bx; bxi++) {
            uint32_t x0 = bxi * FX_BLOCK_SIZE, y0 = byi * FX_BLOCK_SIZE;
            block_info bi = analyze_block(img, w, h, x0, y0,
                                          FX_BLOCK_SIZE, qf);
            bw_put(&bw, bi.mode, 2);
            double step = qstep_base[bi.mode] * qf;
            int32_t pred = predict_level(rec, w, h, x0, y0, step);
            rice_put(&bw, &rc, rice_map_signed(bi.qmean - pred));

            uint32_t xe = x0 + FX_BLOCK_SIZE, ye = y0 + FX_BLOCK_SIZE;
            if (xe > w) xe = w;
            if (ye > h) ye = h;
            for (uint32_t y = y0; y < ye; y++)
                memset(rec + (size_t)y * w + x0, bi.recon, xe - x0);
            nblocks++;
            if (bw.overflow) { fail = 1; break; }
        }
    }
    free(rec);
    if (fail) return -1;
    size_t body = bw_finish(&bw);
    if (body == (size_t)-1) return -1;
    if (out_blocks) *out_blocks = nblocks;
    return (int)(FX_HDR + body);
}

QT_API int fx16_decode(const uint8_t *buf, size_t len,
                       uint8_t *out_img, uint32_t w, uint32_t h)
{
    if (!buf || !out_img || len < FX_HDR || buf[0] != 'F' ||
        buf[1] != '1' || buf[2] != QT_VERSION)
        return -1;
    uint32_t sw = get_u32le(buf + 3), sh = get_u32le(buf + 7);
    if (sw != w || sh != h) return -1;
    double qf = quality_factor(buf[11]);

    bitr br;
    br_init(&br, buf + FX_HDR, len - FX_HDR);
    rice rc;
    rice_init(&rc);

    uint32_t bx = (w + FX_BLOCK_SIZE - 1) / FX_BLOCK_SIZE;
    uint32_t by = (h + FX_BLOCK_SIZE - 1) / FX_BLOCK_SIZE;
    for (uint32_t byi = 0; byi < by; byi++) {
        for (uint32_t bxi = 0; bxi < bx; bxi++) {
            uint32_t x0 = bxi * FX_BLOCK_SIZE, y0 = byi * FX_BLOCK_SIZE;
            uint32_t m = br_get(&br, 2);
            if (m > 3 || br.error) return -1;
            double step = qstep_base[m] * qf;
            int32_t pred = predict_level(out_img, w, h, x0, y0, step);
            int32_t level = pred + rice_unmap_signed(rice_get(&br, &rc));
            uint8_t recon = reconstruct_level(level, step);

            uint32_t xe = x0 + FX_BLOCK_SIZE, ye = y0 + FX_BLOCK_SIZE;
            if (xe > w) xe = w;
            if (ye > h) ye = h;
            for (uint32_t y = y0; y < ye; y++)
                memset(out_img + (size_t)y * w + x0, recon, xe - x0);
        }
    }
    return br.error ? -1 : 0;
}

/* ================================================================== */
/* Demonstration / acceptance harness                                  */
/* ================================================================== */
#ifdef BITGRAIN_QT_DEMO

#include <time.h>

#define IMG_W 256
#define IMG_H 256
#define N_MATERIALS 4
#define TIME_REPEATS 30

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static double psnr_db(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint64_t sse = 0;
    for (size_t i = 0; i < n; i++) {
        int d = (int)a[i] - (int)b[i];
        sse += (uint64_t)d * d;
    }
    if (sse == 0) return 99.0;
    double mse = (double)sse / (double)n;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

/* Deterministic LCG so samples need no stored files. */
static uint32_t lcg_state;
static uint32_t lcg_next(void)
{
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (lcg_state >> 16) & 0x7fff;
}

static void gen_solid(uint8_t *p, size_t n)
{ memset(p, 137, n); }

static void gen_screenshot(uint8_t *p, uint32_t w, uint32_t h)
{
    memset(p, 230, (size_t)w * h);                     /* desktop background */
    for (uint32_t y = 20; y < 220; y++)               /* window body */
        for (uint32_t x = 24; x < 236; x++)
            p[y * w + x] = 245;
    for (uint32_t y = 20; y < 48; y++)                /* title bar */
        for (uint32_t x = 24; x < 236; x++)
            p[y * w + x] = 60;
    for (uint32_t y = 60; y < 80; y++)                /* text lines */
        for (uint32_t x = 40; x < 180; x += 8)
            p[y * w + x] = 30;
    for (uint32_t y = 96; y < 116; y++)
        for (uint32_t x = 40; x < 200; x += 8)
            p[y * w + x] = 30;
    for (uint32_t y = 140; y < 200; y++)              /* solid UI panel A */
        for (uint32_t x = 40; x < 120; x++)
            p[y * w + x] = 200;
    for (uint32_t y = 140; y < 200; y++)              /* solid UI panel B */
        for (uint32_t x = 132; x < 212; x++)
            p[y * w + x] = 180;
}

static void gen_photo(uint8_t *p, uint32_t w, uint32_t h)
{
    lcg_state = 0x1234abcd;
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            double shade = 90.0 + 60.0 * sin((double)x / 13.0) +
                           45.0 * cos((double)y / 17.0);
            double grain = (double)(lcg_next() % 48) - 24.0;
            long v = lround(shade + grain);
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            p[y * w + x] = (uint8_t)v;
        }
    }
}

static void gen_gradient(uint8_t *p, uint32_t w, uint32_t h)
{
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            long v = lround((double)x * 255.0 / (w - 1) * 0.9 +
                            (double)y * 255.0 / (h - 1) * 0.1);
            p[y * w + x] = (uint8_t)v;
        }
}

typedef struct {
    const char *name;
    void (*gen)(uint8_t *, uint32_t, uint32_t);
} material;

int main(int argc, char **argv)
{
    if (argc > 1) qt_lambda_scale = atof(argv[1]);

    static const material mats[N_MATERIALS] = {
        { "solid",      NULL },
        { "screenshot", gen_screenshot },
        { "photo",      gen_photo },
        { "gradient",   gen_gradient },
    };

    const size_t npix = (size_t)IMG_W * IMG_H;
    const size_t cap = 4 * npix + 4096;
    uint8_t *img = (uint8_t *)malloc(npix);
    uint8_t *dec = (uint8_t *)malloc(npix);
    uint8_t *dec_fx = (uint8_t *)malloc(npix);
    uint8_t *buf_qt = (uint8_t *)malloc(cap);
    uint8_t *buf_fx = (uint8_t *)malloc(cap);
    if (!img || !dec || !dec_fx || !buf_qt || !buf_fx) return 2;

    uint64_t total_qt = 0, total_fx = 0;
    int failures = 0;

    printf("bitgrain quadtree partition codec — acceptance run\n");
    printf("image: %ux%u grayscale, quality=%d, max_depth=%u, "
           "var_thresh=%.1f, repeats=%d\n\n",
           IMG_W, IMG_H, QT_QUALITY_DFLT, (unsigned)QT_MAX_DEPTH_CAP,
           QT_VARIANCE_DFLT, TIME_REPEATS);
    printf("%-10s | %-28s | %6s | %8s %8s %6s | %6s %6s | %7s %7s\n",
           "material", "leaves by size 64..4 (depth)", "fx16#",
           "qt_B", "fx_B", "drop%", "qtPSNR", "fxPSNR", "enc_ms", "dec_ms");
    for (int i = 0; i < 118; i++) putchar('-');
    putchar('\n');

    for (int mi = 0; mi < N_MATERIALS; mi++) {
        if (mats[mi].gen) mats[mi].gen(img, IMG_W, IMG_H);
        else gen_solid(img, npix);

        qt_stats st;
        memset(&st, 0, sizeof(st));
        int qt_len = qt_encode(img, IMG_W, IMG_H, QT_QUALITY_DFLT,
                               QT_MAX_DEPTH_CAP, QT_VARIANCE_DFLT,
                               buf_qt, cap, &st);
        uint32_t fx_blocks = 0;
        int fx_len = fx16_encode(img, IMG_W, IMG_H, QT_QUALITY_DFLT,
                                 buf_fx, cap, &fx_blocks);
        if (qt_len < 0 || fx_len < 0) {
            printf("%s: encode failure\n", mats[mi].name);
            failures++;
            continue;
        }

        /* Correctness: decode must reproduce the partition reconstruction
         * and echo the configured max depth from the stream header. */
        memset(dec, 0, npix);
        uint8_t got_depth = 0xFF;
        if (qt_decode(buf_qt, (size_t)qt_len, dec, IMG_W, IMG_H,
                      &got_depth) || got_depth != QT_MAX_DEPTH_CAP) {
            printf("%s: qt decode failure (got_depth=%u)\n",
                   mats[mi].name, got_depth);
            failures++;
        }
        memset(dec_fx, 0, npix);
        if (fx16_decode(buf_fx, (size_t)fx_len, dec_fx, IMG_W, IMG_H)) {
            printf("%s: fx16 decode failure\n", mats[mi].name);
            failures++;
        }

        double p_qt = psnr_db(img, dec, npix);
        double p_fx = psnr_db(img, dec_fx, npix);

        double t0 = now_ms();
        for (int r = 0; r < TIME_REPEATS; r++)
            qt_encode(img, IMG_W, IMG_H, QT_QUALITY_DFLT, QT_MAX_DEPTH_CAP,
                      QT_VARIANCE_DFLT, buf_qt, cap, NULL);
        double enc_ms = (now_ms() - t0) / TIME_REPEATS;

        t0 = now_ms();
        for (int r = 0; r < TIME_REPEATS; r++) {
            memset(dec, 0, npix);
            qt_decode(buf_qt, (size_t)qt_len, dec, IMG_W, IMG_H, NULL);
        }
        double dec_ms = (now_ms() - t0) / TIME_REPEATS;

        uint32_t leaves = 0;
        char dist[96];
        size_t off = 0;
        for (int d = 0; d <= QT_MAX_DEPTH_CAP; d++) {
            leaves += st.leaves_by_depth[d];
            off += (size_t)snprintf(dist + off, sizeof(dist) - off,
                                    d ? " %5u" : "%5u",
                                    st.leaves_by_depth[d]);
        }
        double drop = 100.0 * (1.0 - (double)qt_len / fx_len);
        printf("%-10s | %-28s | %6u | %8d %8d %5.1f%% | %6.2f %6.2f | "
               "%7.3f %7.3f\n",
               mats[mi].name, dist, fx_blocks, qt_len, fx_len, drop,
               p_qt, p_fx, enc_ms, dec_ms);
        total_qt += (uint64_t)qt_len;
        total_fx += (uint64_t)fx_len;

        if (mi == 0) {
            uint32_t theoretical_min =
                ((IMG_W + QT_TILE_SIZE - 1) / QT_TILE_SIZE) *
                ((IMG_H + QT_TILE_SIZE - 1) / QT_TILE_SIZE);
            printf("\n[solid] leaf count: measured=%u  "
                   "theoretical minimum=root tiles=ceil(%u/%u)*ceil(%u/%u)=%u "
                   "(min %s)\n",
                   leaves, IMG_W, QT_TILE_SIZE, IMG_H, QT_TILE_SIZE,
                   theoretical_min,
                   leaves == theoretical_min ? "REACHED" : "NOT reached");
            if (leaves != theoretical_min) failures++;
        }
    }

    double total_drop = 100.0 * (1.0 - (double)total_qt / (double)total_fx);
    for (int i = 0; i < 118; i++) putchar('-');
    putchar('\n');
    printf("aggregate: quadtree=%llu B  fixed16=%llu B  drop=%.2f%% "
           "(target >=15%%: %s)\n\n",
           (unsigned long long)total_qt, (unsigned long long)total_fx,
           total_drop, total_drop >= 15.0 ? "PASS" : "FAIL");
    if (total_drop < 15.0) failures++;

    /* ---- max-depth cap: subdivision must stop at the configured depth ---- */
    printf("max-depth sweep (solid + screenshot), leaf size = 64 >> depth\n");
    printf("%-10s | %-5s | %-32s | %7s | %s\n",
           "material", "depth", "leaves by depth 0..4", "leaves",
           "cap respected");
    for (int mi = 0; mi < 2; mi++) {
        if (mats[mi].gen) mats[mi].gen(img, IMG_W, IMG_H);
        else gen_solid(img, npix);
        for (int md = 0; md <= QT_MAX_DEPTH_CAP; md++) {
            qt_stats st;
            memset(&st, 0, sizeof(st));
            int n = qt_encode(img, IMG_W, IMG_H, QT_QUALITY_DFLT,
                              (uint8_t)md, QT_VARIANCE_DFLT,
                              buf_qt, cap, &st);
            if (n < 0) { failures++; printf("encode failure\n"); continue; }
            uint32_t leaves = 0, deepest = 0;
            char dist[96];
            size_t off = 0;
            for (int d = 0; d <= QT_MAX_DEPTH_CAP; d++) {
                if (st.leaves_by_depth[d]) deepest = (uint32_t)d;
                leaves += st.leaves_by_depth[d];
                off += (size_t)snprintf(dist + off, sizeof(dist) - off,
                                        d ? " %5u" : "%5u",
                                        st.leaves_by_depth[d]);
            }
            int ok = (deepest <= (uint32_t)md);
            if (!ok) failures++;
            printf("%-10s | %-5d | %-32s | %7u | %s (deepest=%u)\n",
                   mats[mi].name, md, dist, leaves,
                   ok ? "yes" : "NO", deepest);
        }
    }

    /* ---- odd dimensions / edge replication round-trip ---- */
    const uint32_t ow = 100, oh = 70;
    uint8_t *oimg = (uint8_t *)malloc(ow * oh);
    uint8_t *odec = (uint8_t *)malloc(ow * oh);
    lcg_state = 7;
    for (size_t i = 0; i < (size_t)ow * oh; i++) oimg[i] = lcg_next() & 255;
    int n1 = qt_encode(oimg, ow, oh, QT_QUALITY_DFLT, 3, 0,
                       buf_qt, cap, NULL);
    int r1 = (n1 > 0) ? qt_decode(buf_qt, (size_t)n1, odec, ow, oh, NULL) : -1;
    int n2 = fx16_encode(oimg, ow, oh, QT_QUALITY_DFLT, buf_fx, cap, NULL);
    int r2 = (n2 > 0) ? fx16_decode(buf_fx, (size_t)n2, odec, ow, oh) : -1;
    printf("\nodd-size %ux%u round-trip: qt=%s fx16=%s\n", ow, oh,
           r1 == 0 ? "ok" : "FAIL", r2 == 0 ? "ok" : "FAIL");
    if (r1 || r2) failures++;
    free(oimg); free(odec);

    free(img); free(dec); free(dec_fx); free(buf_qt); free(buf_fx);
    printf("\n%s\n", failures ? "RESULT: FAIL" : "RESULT: ALL CHECKS PASSED");
    return failures ? 1 : 0;
}
#endif /* BITGRAIN_QT_DEMO */
