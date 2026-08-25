#include "common.h"

// HGRN-Bit (MatMul-Free LM): y = (x_norm @ wq) / scale_w, wq in {-1,0,+1}
// one thread per (o, t) output element, full serial dot product over in_dim
// wq rows are packed 5-trits/byte, matching ggml-quants' TQ1_0 block layout minus its
// per-block fp16 scale (HGRN-Bit already has one scale per whole projection): 256 ternary
// weights -> 52 bytes (48 qs + 4 qh). Ported from matmulfreellmCPU/cpp/include/mmfree/tq1.hpp
// (trit_at / unpack_row); see ggml_compute_forward_hgrn_ternary_mm (ggml-cpu/ops.cpp) for
// the identical CPU-side decode this mirrors.
inline char hgrn_tq1_trit(uchar q, int l) {
    const uchar pow3[5] = { 1, 3, 9, 27, 81 };
    const uchar ql = (uchar) (q * pow3[l]);
    return (char) ((ql >= 86) + (ql >= 171) - 1);
}

inline void hgrn_tq1_unpack_block(thread char * dst, device const uchar * blk) {
    device const uchar * qs = blk;
    device const uchar * qh = blk + 48;
    for (int l = 0; l < 5; l++) {
        for (int j = 0; j < 32; j++) {
            dst[j + 32 * l] = hgrn_tq1_trit(qs[j], l);
        }
    }
    for (int l = 0; l < 5; l++) {
        for (int j = 0; j < 16; j++) {
            dst[160 + j + 16 * l] = hgrn_tq1_trit(qs[32 + j], l);
        }
    }
    for (int l = 0; l < 4; l++) {
        for (int j = 0; j < 4; j++) {
            dst[240 + j + 4 * l] = hgrn_tq1_trit(qh[j], l);
        }
    }
}

// One threadgroup (= one simdgroup, 32 threads) per (o, t) output element: each thread
// strides over a subset of the 256-element blocks (independent accumulators, breaking the
// long serial FMA chain a single-thread reduction would have - see matmulfreellmCPU's
// simd.hpp comment on exactly this problem for the same-shape workload), then simd_sum
// combines the 32 partials. 32 lanes covers every projection's block count in every
// published checkpoint (max 27, at 2.7B's down_proj) with no idle-thread waste beyond that.
// FixedQ510: static signed Q(15-f).f fixed-point activations (default Q5.10), saturating to
// int16, then INTEGER (int32) accumulation over ternary lanes - associative, so bit-exact
// regardless of reduction order (including simd_sum's tree order, unlike the Float path's
// fp32 sum). Dequant by acc / (2^frac_bits * scale_w). Ported from matmulfreellmCPU/cpp/src/
// kernels/bitlinear.cpp (ActQuant::FixedQ510) and mmfree/simd.hpp (quant_q510/
// ternary_dot_i32/dequant_scale) - same math as the CPU/CUDA ports, validated byte-exact
// there against mmfree-cli across all three published checkpoint sizes. rint() rounds to
// nearest, ties to even (Metal Shading Language Spec 6.1 "Common Functions"), matching the
// CPU/CUDA ports' nearbyintf under the default (and essentially universal) FE_TONEAREST mode.
kernel void kernel_hgrn_ternary_mm_f32(
        device const float   * x,
        device const uchar   * wq,
        device const float   * scale,
        device       float   * dst,
        constant     int64_t & in_dim,
        constant     int64_t & out_dim,
        constant     int64_t & n_tok,
        constant     int64_t & row_bytes,
        constant     int     & act_quant,
        constant     int     & frac_bits,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int64_t gid = tgpig.x;
    if (gid >= out_dim * n_tok) {
        return;
    }

    const int64_t o = gid % out_dim;
    const int64_t t = gid / out_dim;
    const int64_t n_blocks = in_dim / 256;

    device const float * xrow    = x  + t * in_dim;
    device const uchar * wpacked = wq + o * row_bytes;

    char wblk[256];

    if (act_quant == 1 /* GGML_HGRN_ACT_QUANT_FIXEDQ510 */) {
        const float qs        = (float) (1 << frac_bits);
        const float inv_fixed = 1.0f / (qs * scale[0]);

        int acc = 0;
        for (int64_t b = tiisg; b < n_blocks; b += 32) {
            hgrn_tq1_unpack_block(wblk, wpacked + b * 52);
            device const float * xblk = xrow + b * 256;
            for (int64_t k = 0; k < 256; k++) {
                float q = rint(xblk[k] * qs);
                q = clamp(q, -32768.0f, 32767.0f);
                const int  yq = (int) q;
                const char wk = wblk[k];
                if (wk > 0) acc += yq;
                else if (wk < 0) acc -= yq;
            }
        }

        acc = simd_sum(acc);

        if (tiisg == 0) {
            dst[t * out_dim + o] = (float) acc * inv_fixed;
        }
        return;
    }

    float acc = 0.0f;
    for (int64_t b = tiisg; b < n_blocks; b += 32) {
        hgrn_tq1_unpack_block(wblk, wpacked + b * 52);
        device const float * xblk = xrow + b * 256;
        for (int64_t k = 0; k < 256; k++) {
            acc += xblk[k] * (float) wblk[k];
        }
    }

    acc = simd_sum(acc);

    if (tiisg == 0) {
        dst[t * out_dim + o] = acc / scale[0];
    }
}

// HGRN-Bit gated linear recurrence: h_t = f_t*h_{t-1} + i_t, y_t = h_t
// state is purely elementwise (no cross-dim mixing), so one thread scans each (d, h, s)
// independently over all T steps; dst = [ y (D*H*T*S) ; new state (D*H*S) ]
kernel void kernel_hgrn_scan_f32(
        device const float   * ii,
        device const float   * ff,
        device const float   * state_in,
        device       float   * dst,
        constant     int64_t & D,
        constant     int64_t & H,
        constant     int64_t & T,
        constant     int64_t & S,
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t gid = tgpig.x;
    if (gid >= D * H * S) {
        return;
    }

    const int64_t d = gid % D;
    const int64_t h = (gid / D) % H;
    const int64_t s = gid / (D * H);

    float state = state_in[(s * H + h) * D + d];

    device float * yout = dst;
    device float * sout = dst + D * H * T * S;

    for (int64_t t = 0; t < T; t++) {
        const int64_t off = ((s * T + t) * H + h) * D + d;
        state = ff[off] * state + ii[off];
        yout[off] = state;
    }

    sout[(s * H + h) * D + d] = state;
}
