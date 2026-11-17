use std::panic::{catch_unwind, AssertUnwindSafe};
use std::cell::{Cell, RefCell};
use std::ffi::CString;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};

extern "C" {
    pub fn quantize_block(
        block: *mut i16,
        table: *const i16,
    );
    pub fn dequantize_block(
        block: *mut i16,
        table: *const i16,
    );
    pub fn bitgrain_dct_block(block: *mut i16);
    pub fn bitgrain_idct_block(block: *mut i16);
}

static RAYON_THREADS_CONFIGURED: AtomicUsize = AtomicUsize::new(0);

const BITGRAIN_OK: i32 = 0;
const BITGRAIN_ERR_INVALID_ARG: i32 = 1;
const BITGRAIN_ERR_DECODE_FAILED: i32 = 2;
const BITGRAIN_ERR_THREAD_INIT: i32 = 3;
const BITGRAIN_ERR_PANIC: i32 = 100;

thread_local! {
    static LAST_ERROR_CODE: Cell<i32> = const { Cell::new(BITGRAIN_OK) };
    static LAST_ERROR_MSG: RefCell<CString> = RefCell::new(CString::new("ok").expect("static cstring"));
}

#[inline]
fn set_last_error(code: i32, msg: &'static str) {
    LAST_ERROR_CODE.with(|c| c.set(code));
    LAST_ERROR_MSG.with(|m| {
        *m.borrow_mut() = CString::new(msg).expect("static cstring");
    });
}

#[inline]
fn clear_last_error() {
    set_last_error(BITGRAIN_OK, "ok");
}

#[inline]
fn fail(code: i32, msg: &'static str) -> i32 {
    set_last_error(code, msg);
    -1
}

#[inline]
fn ffi_guard<F>(f: F) -> i32
where
    F: FnOnce() -> i32,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => fail(BITGRAIN_ERR_PANIC, "panic in codec internals"),
    }
}

#[no_mangle]
pub extern "C" fn bitgrain_last_error_code() -> i32 {
    LAST_ERROR_CODE.with(|c| c.get())
}

#[no_mangle]
pub extern "C" fn bitgrain_last_error_message() -> *const i8 {
    LAST_ERROR_MSG.with(|m| m.borrow().as_c_str().as_ptr())
}

#[no_mangle]
pub extern "C" fn bitgrain_clear_error() {
    clear_last_error();
}

/// Configure global Rayon thread pool size.
/// Must be called before any parallel codec operation starts.
/// Returns 0 on success, -1 on invalid value or late/failed init.
#[no_mangle]
pub extern "C" fn bitgrain_set_threads(threads: i32) -> i32 {
    clear_last_error();
    if threads <= 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "threads must be > 0");
    }
    let t = threads as usize;
    let current = RAYON_THREADS_CONFIGURED.load(Ordering::Relaxed);
    if current == t {
        return 0;
    }
    if current != 0 && current != t {
        return fail(BITGRAIN_ERR_THREAD_INIT, "thread pool already configured with another value");
    }
    match rayon::ThreadPoolBuilder::new().num_threads(t).build_global() {
        Ok(()) => {
            RAYON_THREADS_CONFIGURED.store(t, Ordering::Relaxed);
            0
        }
        Err(_) => fail(BITGRAIN_ERR_THREAD_INIT, "failed to initialize global thread pool"),
    }
}

/// Encode grayscale image.
/// quality: 1–100 (higher = less quantization), 0 = default 85.
#[no_mangle]
pub extern "C" fn bitgrain_encode_grayscale(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_grayscale arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize).saturating_mul(height as usize);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        let mut pos: i32 = 0;
        crate::encoder::encode_grayscale(
            image_slice,
            width as usize,
            height as usize,
            q,
            buffer_slice,
            &mut pos,
        );
        unsafe { *out_len = pos };
        0
    })
}

/// Encode an RGB image (24 bpp, R G B per pixel) to .bg stream.
/// quality: 1–100, 0 = default 85.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgb(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgb arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        let mut pos: i32 = 0;
        if let Some(basis) = wavelet_basis_from_env() {
            if (width as usize) < 2 || (height as usize) < 2 {
                return fail(BITGRAIN_ERR_INVALID_ARG, "wavelet: invalid transform size (need >= 2x2)");
            }
            wavelet_encode(image_slice, width as usize, height as usize, q,
                           buffer_slice, &mut pos, None, false, basis);
        } else {
            crate::encoder::encode_rgb(
                image_slice,
                width as usize,
                height as usize,
                q,
                buffer_slice,
                &mut pos,
                None,
            );
        }
        unsafe { *out_len = pos };
        0
    })
}

/// Encode an RGBA image (32 bpp, R G B A per pixel) to .bg stream.
/// quality: 1–100, 0 = default 85.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgba(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgba arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        let mut pos: i32 = 0;
        if let Some(basis) = wavelet_basis_from_env() {
            if (width as usize) < 2 || (height as usize) < 2 {
                return fail(BITGRAIN_ERR_INVALID_ARG, "wavelet: invalid transform size (need >= 2x2)");
            }
            wavelet_encode(image_slice, width as usize, height as usize, q,
                           buffer_slice, &mut pos, None, true, basis);
        } else {
            crate::encoder::encode_rgba(
                image_slice,
                width as usize,
                height as usize,
                q,
                buffer_slice,
                &mut pos,
                None,
            );
        }
        unsafe { *out_len = pos };
        0
    })
}

/// Decode a .bg stream into pixels (grayscale or RGB per header).
/// out_channels: output, 1 = grayscale (out_pixels = w*h), 3 = RGB (out_pixels = w*h*3).
/// out_capacity debe ser >= width*height*out_channels.
#[no_mangle]
pub extern "C" fn bitgrain_decode(
    buffer: *const u8,
    size: i32,
    out_pixels: *mut u8,
    out_capacity: u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_channels: *mut u32,
) -> i32 {
    clear_last_error();
    if buffer.is_null() || out_pixels.is_null() || out_width.is_null()
        || out_height.is_null() || out_channels.is_null() {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode arguments");
    }
    if size <= 0 || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode buffer size/capacity");
    }
    ffi_guard(|| {
        let buf_slice = unsafe { slice::from_raw_parts(buffer, size as usize) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_pixels, out_capacity as usize) };
        let ok = if is_wavelet_stream(buf_slice) {
            wavelet_decode(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                None,
            )
        } else {
            crate::decoder::decode(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                None,
            )
        };
        if ok {
            0
        } else {
            fail(BITGRAIN_ERR_DECODE_FAILED, "decode failed")
        }
    })
}

/// Decode a .bg stream to grayscale (version 1 .bg only).
#[no_mangle]
pub extern "C" fn bitgrain_decode_grayscale(
    buffer: *const u8,
    size: i32,
    out_pixels: *mut u8,
    out_capacity: u32,
    out_width: *mut u32,
    out_height: *mut u32,
) -> i32 {
    clear_last_error();
    if buffer.is_null() || out_pixels.is_null() || out_width.is_null() || out_height.is_null() {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_grayscale arguments");
    }
    if size <= 0 || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_grayscale buffer size/capacity");
    }
    ffi_guard(|| {
        let mut ch = 0u32;
        let buf_slice = unsafe { slice::from_raw_parts(buffer, size as usize) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_pixels, out_capacity as usize) };
        let ok = crate::decoder::decode(
            buf_slice,
            out_slice,
            unsafe { &mut *out_width },
            unsafe { &mut *out_height },
            &mut ch,
            None,
        );
        if ok && ch == 1 {
            0
        } else {
            fail(BITGRAIN_ERR_DECODE_FAILED, "decode_grayscale failed or output is not grayscale")
        }
    })
}

/// Encode RGB with optional ICC profile.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgb_icc(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
    icc: *const u8,
    icc_len: u32,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgb_icc arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(3);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        let mut pos: i32 = 0;
        let icc_opt = if !icc.is_null() && icc_len > 0 {
            Some(unsafe { slice::from_raw_parts(icc, icc_len as usize) })
        } else {
            None
        };
        if let Some(basis) = wavelet_basis_from_env() {
            if (width as usize) < 2 || (height as usize) < 2 {
                return fail(BITGRAIN_ERR_INVALID_ARG, "wavelet: invalid transform size (need >= 2x2)");
            }
            wavelet_encode(image_slice, width as usize, height as usize, q,
                           buffer_slice, &mut pos, icc_opt, false, basis);
        } else {
            crate::encoder::encode_rgb(
                image_slice,
                width as usize,
                height as usize,
                q,
                buffer_slice,
                &mut pos,
                icc_opt,
            );
        }
        unsafe { *out_len = pos };
        0
    })
}

/// Encode RGBA with optional ICC profile.
#[no_mangle]
pub extern "C" fn bitgrain_encode_rgba_icc(
    image: *const u8,
    width: u32,
    height: u32,
    out_buffer: *mut u8,
    out_capacity: u32,
    out_len: *mut i32,
    quality: u8,
    icc: *const u8,
    icc_len: u32,
) -> i32 {
    clear_last_error();
    if image.is_null() || out_buffer.is_null() || out_len.is_null() || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid encode_rgba_icc arguments");
    }
    ffi_guard(|| {
        let q = if quality == 0 { 85 } else { quality };
        let size = (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(4);
        let image_slice = unsafe { slice::from_raw_parts(image, size) };
        let buffer_slice = unsafe { slice::from_raw_parts_mut(out_buffer, out_capacity as usize) };
        let mut pos: i32 = 0;
        let icc_opt = if !icc.is_null() && icc_len > 0 {
            Some(unsafe { slice::from_raw_parts(icc, icc_len as usize) })
        } else {
            None
        };
        if let Some(basis) = wavelet_basis_from_env() {
            if (width as usize) < 2 || (height as usize) < 2 {
                return fail(BITGRAIN_ERR_INVALID_ARG, "wavelet: invalid transform size (need >= 2x2)");
            }
            wavelet_encode(image_slice, width as usize, height as usize, q,
                           buffer_slice, &mut pos, icc_opt, true, basis);
        } else {
            crate::encoder::encode_rgba(
                image_slice,
                width as usize,
                height as usize,
                q,
                buffer_slice,
                &mut pos,
                icc_opt,
            );
        }
        unsafe { *out_len = pos };
        0
    })
}

/// Decode .bg and optionally return embedded ICC. Caller must free ICC with bitgrain_free_icc.
#[no_mangle]
pub extern "C" fn bitgrain_decode_icc(
    buffer: *const u8,
    size: i32,
    out_pixels: *mut u8,
    out_capacity: u32,
    out_width: *mut u32,
    out_height: *mut u32,
    out_channels: *mut u32,
    out_icc: *mut *mut u8,
    out_icc_len: *mut u32,
) -> i32 {
    clear_last_error();
    if buffer.is_null() || out_pixels.is_null() || out_width.is_null()
        || out_height.is_null() || out_channels.is_null() {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_icc arguments");
    }
    if size <= 0 || out_capacity == 0 {
        return fail(BITGRAIN_ERR_INVALID_ARG, "invalid decode_icc buffer size/capacity");
    }
    ffi_guard(|| {
        let buf_slice = unsafe { slice::from_raw_parts(buffer, size as usize) };
        let out_slice = unsafe { slice::from_raw_parts_mut(out_pixels, out_capacity as usize) };
        let mut icc_vec = Vec::new();
        let ok = if is_wavelet_stream(buf_slice) {
            wavelet_decode(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                Some(&mut icc_vec),
            )
        } else {
            crate::decoder::decode(
                buf_slice,
                out_slice,
                unsafe { &mut *out_width },
                unsafe { &mut *out_height },
                unsafe { &mut *out_channels },
                Some(&mut icc_vec),
            )
        };
        if !ok {
            return fail(BITGRAIN_ERR_DECODE_FAILED, "decode_icc failed");
        }
        if !out_icc.is_null() && !out_icc_len.is_null() {
            if icc_vec.is_empty() {
                unsafe {
                    *out_icc = std::ptr::null_mut();
                    *out_icc_len = 0;
                }
            } else {
                let len = icc_vec.len();
                let ptr = icc_vec.as_mut_ptr();
                std::mem::forget(icc_vec);
                unsafe {
                    *out_icc = ptr;
                    *out_icc_len = len as u32;
                }
            }
        }
        0
    })
}

/// Free ICC buffer returned by bitgrain_decode_icc.
#[no_mangle]
pub extern "C" fn bitgrain_free_icc(ptr: *mut u8, len: u32) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let _ = unsafe { Vec::from_raw_parts(ptr, len as usize, len as usize) };
}

// ===========================================================================
// Wavelet transform-domain codec (opt-in experiment, .bg versions 20/21)
//
// Same 12-byte .bg container/header; new version bytes select the transform:
//   20 = YCbCr 4:2:0, one-level 2D integer wavelet on luma (RGB output)
//   21 = same + full-res alpha plane (RGBA output)
//
// Luma plane: one-level 2D wavelet (Haar or CDF 5/3, both integer and
// exactly reversible). The four subbands (LL, LH, HL, HH) are each uniformly
// quantized with a quality-scaled step and entropy-coded separately with the
// existing JPEG-style Huffman plane coder (8x8 tiling, DC delta). Chroma and
// alpha planes reuse the v18/v19 DCT pipeline (mirrored locally so the
// spatial default keeps bit-identical output in encoder.rs).
//
// Opt-in via environment (public C API signatures unchanged):
//   BITGRAIN_WAVELET=haar|cdf53   select basis; unset = spatial DCT (default)
//   BITGRAIN_WAVELET_STATS=1      print luma subband energy split to stderr
//
// Transform size validity: a one-level 2D wavelet needs >= 2 samples per
// dimension to split into four subbands, so width < 2 or height < 2 is
// rejected at encode (BITGRAIN_ERR_INVALID_ARG) and at decode (failure).
// Odd dimensions are legal (symmetric-extension lifting).
// ===========================================================================

const BG_VERSION_WAVELET_RGB: u8 = 20;
const BG_VERSION_WAVELET_RGBA: u8 = 21;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WaveletBasis {
    Haar,
    Cdf53,
}

impl WaveletBasis {
    fn tag(self) -> u8 {
        match self {
            WaveletBasis::Haar => 0,
            WaveletBasis::Cdf53 => 1,
        }
    }
    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(WaveletBasis::Haar),
            1 => Some(WaveletBasis::Cdf53),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            WaveletBasis::Haar => "haar",
            WaveletBasis::Cdf53 => "cdf53",
        }
    }
}

fn wavelet_basis_from_env() -> Option<WaveletBasis> {
    match std::env::var("BITGRAIN_WAVELET").ok().as_deref() {
        Some("haar") => Some(WaveletBasis::Haar),
        Some("cdf53") | Some("5/3") => Some(WaveletBasis::Cdf53),
        _ => None,
    }
}

fn wavelet_stats_enabled() -> bool {
    matches!(std::env::var("BITGRAIN_WAVELET_STATS").ok().as_deref(), Some("1"))
}

fn is_wavelet_stream(buf: &[u8]) -> bool {
    buf.len() >= 3 && (buf[2] == BG_VERSION_WAVELET_RGB || buf[2] == BG_VERSION_WAVELET_RGBA)
}

// ---------------------------------------------------------------------------
// 1D lifting (in-place, output layout [lows..., highs...])
// Requires n >= 2. Symmetric whole-sample extension at boundaries.
// ---------------------------------------------------------------------------

fn wavelet_lift_fwd(line: &mut [i32], basis: WaveletBasis) {
    let n = line.len();
    debug_assert!(n >= 2);
    let nl = (n + 1) / 2;
    let nh = n / 2;
    let src: Vec<i32> = line.to_vec();
    match basis {
        WaveletBasis::Haar => {
            // Reversible Haar: d = b - a; s = a + floor(d/2).
            for i in 0..nh {
                let a = src[2 * i];
                let b = src[2 * i + 1];
                let d = b - a;
                line[i] = a + (d >> 1);
                line[nl + i] = d;
            }
            if n % 2 == 1 {
                line[nl - 1] = src[n - 1];
            }
        }
        WaveletBasis::Cdf53 => {
            // JPEG 2000 reversible 5/3:
            //   d_i = x[2i+1] - floor((x[2i] + x[2i+2]) / 2)
            //   s_i = x[2i]   - floor((-d_{i-1} - d_i - 2) / 4)
            let mut d = vec![0i32; nh.max(1)];
            for i in 0..nh {
                let l = src[2 * i];
                let r = src[if 2 * i + 2 < n { 2 * i + 2 } else { n - 2 }];
                d[i] = src[2 * i + 1] - ((l + r) >> 1);
            }
            for i in 0..nl {
                let dm = if i == 0 { d[0] } else { d[i - 1] };
                let di = if i < nh { d[i] } else { d[nh - 1] };
                line[i] = src[2 * i] + ((dm + di + 2) >> 2);
            }
            for i in 0..nh {
                line[nl + i] = d[i];
            }
        }
    }
}

fn wavelet_lift_inv(line: &mut [i32], basis: WaveletBasis) {
    let n = line.len();
    debug_assert!(n >= 2);
    let nl = (n + 1) / 2;
    let nh = n / 2;
    let ls: Vec<i32> = line[..nl].to_vec();
    let hs: Vec<i32> = line[nl..].to_vec();
    match basis {
        WaveletBasis::Haar => {
            for i in 0..nh {
                let d = hs[i];
                let a = ls[i] - (d >> 1);
                line[2 * i] = a;
                line[2 * i + 1] = a + d;
            }
            if n % 2 == 1 {
                line[n - 1] = ls[nl - 1];
            }
        }
        WaveletBasis::Cdf53 => {
            let mut e = vec![0i32; nl];
            for i in 0..nl {
                let dm = if i == 0 { hs[0] } else { hs[i - 1] };
                let di = if i < nh { hs[i] } else { hs[nh - 1] };
                e[i] = ls[i] - ((dm + di + 2) >> 2);
            }
            for i in 0..nh {
                let r = if 2 * i + 2 < n { e[i + 1] } else { e[(n - 2) / 2] };
                line[2 * i + 1] = hs[i] + ((e[i] + r) >> 1);
            }
            for i in 0..nl {
                line[2 * i] = e[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One-level 2D wavelet on a w*h i32 plane
// ---------------------------------------------------------------------------

fn wavelet_fwd_2d(plane: &mut [i32], w: usize, h: usize, basis: WaveletBasis) {
    for row in plane.chunks_exact_mut(w) {
        wavelet_lift_fwd(row, basis);
    }
    let mut col = vec![0i32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = plane[y * w + x];
        }
        wavelet_lift_fwd(&mut col, basis);
        for y in 0..h {
            plane[y * w + x] = col[y];
        }
    }
}

fn wavelet_inv_2d(plane: &mut [i32], w: usize, h: usize, basis: WaveletBasis) {
    let mut col = vec![0i32; h];
    for x in 0..w {
        for y in 0..h {
            col[y] = plane[y * w + x];
        }
        wavelet_lift_inv(&mut col, basis);
        for y in 0..h {
            plane[y * w + x] = col[y];
        }
    }
    for row in plane.chunks_exact_mut(w) {
        wavelet_lift_inv(row, basis);
    }
}

/// Subband quadrants after the 2D transform. Order: LL, LH, HL, HH
/// (first letter = vertical filter, second = horizontal filter).
struct WaveletSubband {
    coeffs: Vec<i32>,
    w: usize,
    h: usize,
    energy: f64,
}

fn wavelet_subband_dims(w: usize, h: usize) -> [(usize, usize, usize, usize); 4] {
    // (width, height, origin_x, origin_y) for LL, LH, HL, HH
    let lw = (w + 1) / 2;
    let lh = (h + 1) / 2;
    [
        (lw, lh, 0, 0),
        (w - lw, lh, lw, 0),
        (lw, h - lh, 0, lh),
        (w - lw, h - lh, lw, lh),
    ]
}

fn wavelet_extract(plane: &[i32], w: usize, h: usize) -> [WaveletSubband; 4] {
    let dims = wavelet_subband_dims(w, h);
    dims.map(|(sw, sh, ox, oy)| {
        let mut coeffs = vec![0i32; sw * sh];
        let mut energy = 0.0f64;
        for sy in 0..sh {
            let src_row = (oy + sy) * w + ox;
            for sx in 0..sw {
                let c = plane[src_row + sx];
                coeffs[sy * sw + sx] = c;
                energy += (c as f64) * (c as f64);
            }
        }
        WaveletSubband { coeffs, w: sw, h: sh, energy }
    })
}

/// Per-subband uniform quantizer steps, quality-scaled with the same
/// formula as the JPEG tables (quality 50 = base steps, 100 = step 1).
/// Order: LL, LH, HL, HH.
fn wavelet_quant_steps(quality: u8) -> [i32; 4] {
    let q = quality.clamp(1, 100) as i32;
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    const BASE: [i32; 4] = [8, 20, 20, 32];
    let mut out = [0i32; 4];
    for i in 0..4 {
        out[i] = ((BASE[i] * scale + 50) / 100).clamp(1, 255);
    }
    out
}

// ---------------------------------------------------------------------------
// Subband <-> Huffman plane (8x8 tiling, zero-padded edges, DC delta)
// ---------------------------------------------------------------------------

fn wavelet_encode_subband(sb: &WaveletSubband, step: i32) -> Vec<u8> {
    let bw = (sb.w + 7) / 8;
    let bh = (sb.h + 7) / 8;
    let mut blocks = vec![crate::block::Block::new(); bw * bh];
    for by in 0..bh {
        for bx in 0..bw {
            let mut b = crate::block::Block::new();
            for y in 0..8 {
                for x in 0..8 {
                    let sy = by * 8 + y;
                    let sx = bx * 8 + x;
                    if sy < sb.h && sx < sb.w {
                        let c = sb.coeffs[sy * sb.w + sx];
                        let q = if c >= 0 {
                            (c + step / 2) / step
                        } else {
                            -((-c + step / 2) / step)
                        };
                        b.data[y * 8 + x] = q as i16;
                    }
                }
            }
            crate::huffman::clamp_block_jpeg_coeffs(&mut b);
            blocks[by * bw + bx] = b;
        }
    }
    crate::huffman::encode_plane_with_profile(&blocks, false, false, true)
}

fn wavelet_decode_subband(
    buf: &[u8],
    pos: usize,
    sw: usize,
    sh: usize,
    step: i32,
) -> Option<(Vec<i32>, usize)> {
    let bw = (sw + 7) / 8;
    let n = bw * ((sh + 7) / 8);
    let (blocks, new_pos) =
        crate::huffman::decode_plane_with_profile(buf, pos, n, false, false, true)?;
    let mut coeffs = vec![0i32; sw * sh];
    for (idx, b) in blocks.iter().enumerate() {
        let bx = (idx % bw) * 8;
        let by = (idx / bw) * 8;
        for y in 0..8 {
            for x in 0..8 {
                let sy = by + y;
                let sx = bx + x;
                if sy < sh && sx < sw {
                    coeffs[sy * sw + sx] = b.data[y * 8 + x] as i32 * step;
                }
            }
        }
    }
    Some((coeffs, new_pos))
}

// ---------------------------------------------------------------------------
// Chroma/alpha planes: local mirror of the v18/v19 DCT pipeline
// (perceptual-v4 tables + AC sparsify + chroma AC table + DC delta)
// ---------------------------------------------------------------------------

/// Quantize wrapper: release builds use the C SIMD kernel; test builds use a
/// pure-Rust equivalent so `cargo test` links without the C objects (same
/// pattern as dct.rs).
#[inline]
fn wavelet_quantize_block(block: &mut [i16; 64], table: &[i16; 64]) {
    #[cfg(not(test))]
    unsafe { quantize_block(block.as_mut_ptr(), table.as_ptr()) }
    #[cfg(test)]
    for i in 0..64 {
        block[i] = (block[i] as f32 / table[i] as f32).round() as i16;
    }
}

#[inline]
fn wavelet_dequantize_block(block: &mut [i16; 64], table: &[i16; 64]) {
    #[cfg(not(test))]
    unsafe { dequantize_block(block.as_mut_ptr(), table.as_ptr()) }
    #[cfg(test)]
    for i in 0..64 {
        block[i] = (block[i] as i32 * table[i] as i32).clamp(-32768, 32767) as i16;
    }
}

fn wavelet_sparsify_thresholds(quality: u8, is_chroma: bool) -> [i16; 64] {
    // Mirror of encoder.rs::build_sparsify_thresholds (v18/v19 profile).
    let q = quality.clamp(1, 100);
    let base_t: i16 = if is_chroma {
        if q >= 90 { 2 } else if q >= 80 { 3 } else { 4 }
    } else if q >= 90 {
        1
    } else if q >= 80 {
        2
    } else {
        3
    };
    let mut thr = [0i16; 64];
    for zi in 1..64 {
        thr[zi] = if zi >= 56 {
            base_t + 2
        } else if zi >= 40 {
            base_t + 1
        } else if zi >= 24 {
            base_t
        } else {
            base_t.saturating_sub(1).max(1)
        };
    }
    thr
}

fn wavelet_sparsify_ac(block: &mut crate::block::Block, thresholds: &[i16; 64]) {
    for zi in 1..64 {
        let z = crate::zigzag::ZIGZAG[zi];
        if block.data[z].abs() <= thresholds[zi] {
            block.data[z] = 0;
        }
    }
}

fn wavelet_dct_encode_plane(
    plane: &[u8],
    w: usize,
    h: usize,
    table: &[i16; 64],
    is_chroma: bool,
    sparsify: &[i16; 64],
) -> Vec<u8> {
    let blockizer = crate::blockizer::Blockizer::new(w, h);
    let mut blocks = blockizer.generate_blocks(plane);
    for b in blocks.iter_mut() {
        crate::dct::dct(b);
        wavelet_quantize_block(&mut b.data, table);
        wavelet_sparsify_ac(b, sparsify);
        crate::huffman::clamp_block_jpeg_coeffs(b);
    }
    crate::huffman::encode_plane_with_profile(&blocks, is_chroma, is_chroma, true)
}

fn wavelet_dct_decode_plane(
    buf: &[u8],
    pos: usize,
    w: usize,
    h: usize,
    table: &[i16; 64],
    is_chroma: bool,
    plane: &mut [u8],
) -> Option<usize> {
    let bw = (w + 7) / 8;
    let n = bw * ((h + 7) / 8);
    let (mut blocks, new_pos) =
        crate::huffman::decode_plane_with_profile(buf, pos, n, is_chroma, is_chroma, true)?;
    for b in blocks.iter_mut() {
        wavelet_dequantize_block(&mut b.data, table);
        crate::dct::idct(b);
    }
    for (idx, b) in blocks.iter().enumerate() {
        let bx = (idx % bw) * 8;
        let by = (idx / bw) * 8;
        for y in 0..8 {
            for x in 0..8 {
                let py = by + y;
                let px = bx + x;
                if py < h && px < w {
                    plane[py * w + px] = (b.data[y * 8 + x] + 128).clamp(0, 255) as u8;
                }
            }
        }
    }
    Some(new_pos)
}

// ---------------------------------------------------------------------------
// Container helpers (same 12-byte header and ICC trailer as the DCT path)
// ---------------------------------------------------------------------------

fn wavelet_write_header(out: &mut [u8], pos: &mut i32, version: u8, w: usize, h: usize, q: u8) {
    crate::bitstream::write_bytes(out, pos, &[b'B', b'G', version]);
    for b in (w as u32).to_le_bytes() {
        crate::bitstream::write_byte(out, pos, b);
    }
    for b in (h as u32).to_le_bytes() {
        crate::bitstream::write_byte(out, pos, b);
    }
    crate::bitstream::write_byte(out, pos, if q == 0 { 50 } else { q });
}

fn wavelet_write_icc_trailer(out: &mut [u8], pos: &mut i32, icc: Option<&[u8]>) {
    let Some(data) = icc else { return };
    if data.is_empty() {
        return;
    }
    crate::bitstream::write_byte(out, pos, b'B');
    crate::bitstream::write_byte(out, pos, b'G');
    crate::bitstream::write_byte(out, pos, b'x');
    crate::bitstream::write_byte(out, pos, 1);
    for b in (data.len() as u32).to_le_bytes() {
        crate::bitstream::write_byte(out, pos, b);
    }
    crate::bitstream::write_bytes(out, pos, data);
}

fn wavelet_parse_icc(buffer: &[u8], pos: usize) -> Option<(Vec<u8>, usize)> {
    if pos + 8 > buffer.len() {
        return None;
    }
    if buffer[pos] != b'B' || buffer[pos + 1] != b'G' || buffer[pos + 2] != b'x' {
        return None;
    }
    let chunk_type = buffer[pos + 3];
    let len = u32::from_le_bytes(buffer[pos + 4..pos + 8].try_into().unwrap()) as usize;
    let data_pos = pos + 8;
    if chunk_type != 1 || data_pos + len > buffer.len() {
        return None;
    }
    Some((buffer[data_pos..data_pos + len].to_vec(), data_pos + len))
}

// ---------------------------------------------------------------------------
// Top-level wavelet encode / decode
// ---------------------------------------------------------------------------

/// Encode RGB (`rgba = false`, v20) or RGBA (`rgba = true`, v21) with a
/// one-level 2D wavelet on the luma plane. Caller guarantees w,h >= 2.
fn wavelet_encode(
    image: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    out: &mut [u8],
    pos: &mut i32,
    icc: Option<&[u8]>,
    rgba: bool,
    basis: WaveletBasis,
) {
    let version = if rgba { BG_VERSION_WAVELET_RGBA } else { BG_VERSION_WAVELET_RGB };
    wavelet_write_header(out, pos, version, width, height, quality);
    crate::bitstream::write_byte(out, pos, basis.tag());

    let (y, cb, cr, alpha) = if rgba {
        let (y, cb, cr, a) = crate::colorspace::rgba_to_ycbcr420a(image, width, height);
        (y, cb, cr, Some(a))
    } else {
        let (y, cb, cr) = crate::colorspace::rgb_to_ycbcr420(image, width, height);
        (y, cb, cr, None)
    };
    let cw = (width + 1) / 2;
    let ch = (height + 1) / 2;

    // Luma: one-level 2D wavelet, four subbands quantized + coded separately.
    let mut ybuf = vec![0i32; width * height];
    for (i, &v) in y.iter().enumerate() {
        ybuf[i] = v as i32 - 128;
    }
    wavelet_fwd_2d(&mut ybuf, width, height, basis);
    let subbands = wavelet_extract(&ybuf, width, height);
    let steps = wavelet_quant_steps(quality);

    if wavelet_stats_enabled() {
        let total: f64 = subbands.iter().map(|s| s.energy).sum();
        let pct = |e: f64| if total > 0.0 { 100.0 * e / total } else { 0.0 };
        eprintln!(
            "bitgrain wavelet: basis={} quality={} size={}x{} steps=LL:{} LH:{} HL:{} HH:{} energy LL={:.2}% LH={:.2}% HL={:.2}% HH={:.2}%",
            basis.name(), quality, width, height,
            steps[0], steps[1], steps[2], steps[3],
            pct(subbands[0].energy), pct(subbands[1].energy),
            pct(subbands[2].energy), pct(subbands[3].energy),
        );
    }

    for (i, sb) in subbands.iter().enumerate() {
        let plane = wavelet_encode_subband(sb, steps[i]);
        crate::bitstream::write_bytes(out, pos, &plane);
    }

    // Chroma (and alpha): identical DCT pipeline to the v18/v19 spatial path.
    let chroma_table = crate::encoder::chroma_quant_table_for_quality_perceptual_v4(quality);
    let chroma_sparsify = wavelet_sparsify_thresholds(quality, true);
    let cb_buf = wavelet_dct_encode_plane(&cb, cw, ch, &chroma_table, true, &chroma_sparsify);
    let cr_buf = wavelet_dct_encode_plane(&cr, cw, ch, &chroma_table, true, &chroma_sparsify);
    crate::bitstream::write_bytes(out, pos, &cb_buf);
    crate::bitstream::write_bytes(out, pos, &cr_buf);
    if let Some(a) = &alpha {
        let luma_table = crate::encoder::quant_table_for_quality_perceptual_v4(quality);
        let luma_sparsify = wavelet_sparsify_thresholds(quality, false);
        let a_buf = wavelet_dct_encode_plane(a, width, height, &luma_table, false, &luma_sparsify);
        crate::bitstream::write_bytes(out, pos, &a_buf);
    }
    wavelet_write_icc_trailer(out, pos, icc);
}

/// Decode a v20/v21 wavelet stream. Returns false on any format error,
/// including illegal transform sizes (width < 2 or height < 2).
fn wavelet_decode(
    buffer: &[u8],
    out_pixels: &mut [u8],
    out_width: &mut u32,
    out_height: &mut u32,
    out_channels: &mut u32,
    out_icc: Option<&mut Vec<u8>>,
) -> bool {
    const HEADER: usize = 3 + 4 + 4 + 1;
    if buffer.len() < HEADER + 1 {
        return false;
    }
    if buffer[0] != b'B' || buffer[1] != b'G' {
        return false;
    }
    let rgba = match buffer[2] {
        BG_VERSION_WAVELET_RGB => false,
        BG_VERSION_WAVELET_RGBA => true,
        _ => return false,
    };
    let width = u32::from_le_bytes(buffer[3..7].try_into().unwrap());
    let height = u32::from_le_bytes(buffer[7..11].try_into().unwrap());
    let quality = buffer[11];
    let q = if quality == 0 { 50 } else { quality };
    // A one-level 2D wavelet needs >= 2 samples per dimension.
    if width < 2 || height < 2 || width > 65536 || height > 65536 {
        return false;
    }
    let basis = match WaveletBasis::from_tag(buffer[HEADER]) {
        Some(b) => b,
        None => return false,
    };
    let w = width as usize;
    let h = height as usize;
    let channels: usize = if rgba { 4 } else { 3 };
    if out_pixels.len() < w * h * channels {
        return false;
    }

    // Luma subbands.
    let steps = wavelet_quant_steps(q);
    let dims = wavelet_subband_dims(w, h);
    let mut ybuf = vec![0i32; w * h];
    let mut pos = HEADER + 1;
    for (i, &(sw, sh, ox, oy)) in dims.iter().enumerate() {
        let (coeffs, new_pos) = match wavelet_decode_subband(buffer, pos, sw, sh, steps[i]) {
            Some(v) => v,
            None => return false,
        };
        pos = new_pos;
        for sy in 0..sh {
            let dst_row = (oy + sy) * w + ox;
            for sx in 0..sw {
                ybuf[dst_row + sx] = coeffs[sy * sw + sx];
            }
        }
    }
    wavelet_inv_2d(&mut ybuf, w, h, basis);
    let mut y_plane = vec![0u8; w * h];
    for (i, &c) in ybuf.iter().enumerate() {
        y_plane[i] = (c + 128).clamp(0, 255) as u8;
    }

    // Chroma (and alpha) DCT planes.
    let cw = (w + 1) / 2;
    let ch = (h + 1) / 2;
    let chroma_table = crate::encoder::chroma_quant_table_for_quality_perceptual_v4(q);
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];
    pos = match wavelet_dct_decode_plane(buffer, pos, cw, ch, &chroma_table, true, &mut cb_plane) {
        Some(p) => p,
        None => return false,
    };
    pos = match wavelet_dct_decode_plane(buffer, pos, cw, ch, &chroma_table, true, &mut cr_plane) {
        Some(p) => p,
        None => return false,
    };

    *out_width = width;
    *out_height = height;
    *out_channels = channels as u32;
    if rgba {
        let luma_table = crate::encoder::quant_table_for_quality_perceptual_v4(q);
        let mut a_plane = vec![0u8; w * h];
        pos = match wavelet_dct_decode_plane(buffer, pos, w, h, &luma_table, false, &mut a_plane) {
            Some(p) => p,
            None => return false,
        };
        crate::colorspace::ycbcr420a_to_rgba(&y_plane, &cb_plane, &cr_plane, &a_plane, w, h, out_pixels);
    } else {
        crate::colorspace::ycbcr420_to_rgb(&y_plane, &cb_plane, &cr_plane, w, h, out_pixels);
    }

    if let Some(v) = out_icc {
        if let Some((icc, _)) = wavelet_parse_icc(buffer, pos) {
            *v = icc;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod wavelet_tests {
    use super::*;

    fn roundtrip_1d(basis: WaveletBasis, data: &[i32]) {
        let mut line = data.to_vec();
        wavelet_lift_fwd(&mut line, basis);
        wavelet_lift_inv(&mut line, basis);
        assert_eq!(&line, data, "{basis:?} 1D lifting not reversible");
    }

    #[test]
    fn lift_1d_reversible_both_bases() {
        let cases: Vec<Vec<i32>> = vec![
            vec![0, 0],
            vec![5, -5],
            vec![1, 2, 3],
            vec![-128, 127, -128, 127],
            vec![7, 7, 7, 7, 7],
            (0..64).map(|i| (i * 37 % 255) as i32 - 128).collect(),
            (0..63).map(|i| (i * 91 % 251) as i32 - 125).collect(),
        ];
        for c in &cases {
            roundtrip_1d(WaveletBasis::Haar, c);
            roundtrip_1d(WaveletBasis::Cdf53, c);
        }
    }

    #[test]
    fn transform_2d_reversible_odd_and_even() {
        for &(w, h) in &[(2, 2), (8, 8), (16, 16), (13, 7), (127, 63), (2, 97)] {
            let mut plane: Vec<i32> = (0..w * h).map(|i| (i * 53 % 241) as i32 - 120).collect();
            let orig = plane.clone();
            for basis in [WaveletBasis::Haar, WaveletBasis::Cdf53] {
                wavelet_fwd_2d(&mut plane, w, h, basis);
                wavelet_inv_2d(&mut plane, w, h, basis);
                assert_eq!(plane, orig, "{basis:?} 2D not reversible at {w}x{h}");
            }
        }
    }

    #[test]
    fn quant_steps_quality_monotonic() {
        let lo = wavelet_quant_steps(10);
        let mid = wavelet_quant_steps(50);
        let hi = wavelet_quant_steps(100);
        for i in 0..4 {
            assert!(lo[i] >= mid[i] && mid[i] >= hi[i]);
            assert!(hi[i] >= 1);
        }
    }

    fn sample_rgb(w: usize, h: usize) -> Vec<u8> {
        // R=G=B so Cb/Cr are constant: 4:2:0 chroma subsampling is lossless
        // and the round-trip error isolates the luma (wavelet) path.
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let g = ((x * 2 + y * 3) % 256) as u8;
                v.push(g);
                v.push(g);
                v.push(g);
            }
        }
        v
    }

    fn wavelet_roundtrip(w: usize, h: usize, quality: u8, basis: WaveletBasis) -> (Vec<u8>, Vec<u8>) {
        let img = sample_rgb(w, h);
        let mut buf = vec![0u8; w * h * 3 * 2 + 4096];
        let mut pos: i32 = 0;
        wavelet_encode(&img, w, h, quality, &mut buf, &mut pos, None, false, basis);
        let stream = &buf[..pos as usize];
        assert_eq!(stream[2], BG_VERSION_WAVELET_RGB);
        let mut out = vec![0u8; w * h * 3];
        let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
        assert!(wavelet_decode(stream, &mut out, &mut ow, &mut oh, &mut oc, None));
        assert_eq!((ow, oh, oc), (w as u32, h as u32, 3));
        (img, out)
    }

    #[test]
    fn wavelet_encode_decode_roundtrip_rgb() {
        for basis in [WaveletBasis::Haar, WaveletBasis::Cdf53] {
            for &(w, h) in &[(16, 16), (64, 48), (63, 33)] {
                let (src, dec) = wavelet_roundtrip(w, h, 95, basis);
                // Lossy but close at q95: check mean absolute error is small.
                let mae: f64 = src.iter().zip(dec.iter())
                    .map(|(&a, &b)| (a as f64 - b as f64).abs())
                    .sum::<f64>() / src.len() as f64;
                assert!(mae < 8.0, "{basis:?} {w}x{h} mae={mae}");
            }
        }
    }

    #[test]
    fn wavelet_lossless_at_quality_100_luma_exact() {
        // At q=100 all subband steps are 1, so the integer wavelet is exactly
        // reversible; residual error comes only from colorspace/chroma.
        for basis in [WaveletBasis::Haar, WaveletBasis::Cdf53] {
            let (src, dec) = wavelet_roundtrip(32, 32, 100, basis);
            let max_err = src.iter().zip(dec.iter())
                .map(|(&a, &b)| (a as i32 - b as i32).abs())
                .max().unwrap();
            assert!(max_err <= 12, "{basis:?} max_err={max_err}");
        }
    }

    #[test]
    fn wavelet_decode_rejects_invalid_transform_size() {
        for &(w, h) in &[(1u32, 1u32), (1, 16), (16, 1), (0, 8)] {
            let mut stream = Vec::new();
            stream.extend_from_slice(b"BG");
            stream.push(BG_VERSION_WAVELET_RGB);
            stream.extend_from_slice(&w.to_le_bytes());
            stream.extend_from_slice(&h.to_le_bytes());
            stream.push(85); // quality
            stream.push(0); // basis = haar
            stream.extend_from_slice(&[0u8; 32]);
            let mut out = vec![0u8; 4096];
            let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
            assert!(
                !wavelet_decode(&stream, &mut out, &mut ow, &mut oh, &mut oc, None),
                "accepted illegal transform size {w}x{h}"
            );
        }
    }

    #[test]
    fn wavelet_decode_rejects_bad_basis_tag() {
        let img = sample_rgb(16, 16);
        let mut buf = vec![0u8; 16 * 16 * 3 * 2 + 256];
        let mut pos: i32 = 0;
        wavelet_encode(&img, 16, 16, 85, &mut buf, &mut pos, None, false, WaveletBasis::Haar);
        let mut stream = buf[..pos as usize].to_vec();
        stream[12] = 7; // invalid basis tag
        let mut out = vec![0u8; 16 * 16 * 3];
        let (mut ow, mut oh, mut oc) = (0u32, 0u32, 0u32);
        assert!(!wavelet_decode(&stream, &mut out, &mut ow, &mut oh, &mut oc, None));
    }

    #[test]
    fn wavelet_subband_energy_partition() {
        // Flat plane: all energy must land in LL.
        let (w, h) = (32, 32);
        let mut plane = vec![42i32; w * h];
        wavelet_fwd_2d(&mut plane, w, h, WaveletBasis::Haar);
        let sbs = wavelet_extract(&plane, w, h);
        assert!(sbs[0].energy > 0.0);
        for sb in &sbs[1..] {
            assert_eq!(sb.energy, 0.0);
        }
    }
}
