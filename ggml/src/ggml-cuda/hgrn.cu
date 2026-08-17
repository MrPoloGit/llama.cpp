#include "common.cuh"
#include "hgrn.cuh"

// wq rows are packed 5-trits/byte, matching ggml-quants' TQ1_0 block layout minus its
// per-block fp16 scale (HGRN-Bit already has one scale per whole projection): 256 ternary
// weights -> 52 bytes (48 qs + 4 qh). Ported from matmulfreellmCPU/cpp/include/mmfree/tq1.hpp
// (trit_at / unpack_row); see ggml_compute_forward_hgrn_ternary_mm (ggml-cpu/ops.cpp) for the
// identical CPU-side decode this mirrors.
__device__ __forceinline__ int8_t hgrn_tq1_trit(uint8_t q, int l) {
    const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    const uint8_t ql = (uint8_t) (q * pow3[l]);
    return (int8_t) ((ql >= 86) + (ql >= 171) - 1);
}

__device__ __forceinline__ void hgrn_tq1_unpack_block(int8_t * dst, const uint8_t * blk) {
    const uint8_t * qs = blk;
    const uint8_t * qh = blk + 48;
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

// HGRN-Bit (MatMul-Free LM) ternary BitLinear: y = (x_norm @ wq) / scale_w, wq in {-1,0,+1}
// one thread per (o, t) output element, full serial dot product over in_dim (correctness-first,
// same math as the CPU/Metal ports - see src/models/hgrnbit.cpp)
//
// Two cooperative-reduction variants were tried and measured *slower* than this plain serial
// kernel on a GTX 1050 Ti (sm_61, 4GB): a fixed 32-lane warp per output element (3-3.5x
// slower - most lanes idle, since every published checkpoint's block count, in_dim/256, is
// only 4-27), and a version sized to the exact block count with zero idle lanes (still
// 1.4-2.2x slower, e.g. 1.3B's 8-block projections at a perfectly-matched GROUP=8). The
// second result rules out "wasted lanes" as the whole story: even fully-utilized cooperative
// threads lose here, most likely because every additional concurrently-resident thread still
// carries its own copy of this function's 256-byte wblk scratch buffer, and that per-thread
// register/local-memory cost outweighs whatever serial-chain time is saved on a
// register-constrained card, for chains this short (4-27 iterations). Conclusion for this
// hardware: no cooperative reduction at all is the fastest option. A GPU with a much larger
// register file per SM (a modern Ampere+/workstation card, not this old 4GB Pascal laptop
// part) might have different economics - revisit only with real before/after numbers on the
// target GPU, not by analogy to another backend or another kernel design.
// FixedQ510: static signed Q(15-f).f fixed-point activations (default Q5.10), saturating to
// int16, then INTEGER (int32) accumulation over ternary lanes - associative, so bit-exact
// regardless of reduction order. Dequant by acc / (2^frac_bits * scale_w). Ported from
// matmulfreellmCPU/cpp/src/kernels/bitlinear.cpp (ActQuant::FixedQ510) and mmfree/simd.hpp
// (quant_q510/ternary_dot_i32/dequant_scale) - same math as the CPU port (ggml-cpu/ops.cpp),
// validated byte-exact there against mmfree-cli across all three published checkpoint sizes.
// nearbyintf is a real CUDA device math function (round-to-nearest-even, matching host libm),
// not an approximation - see the CUDA Math API guide.
static __global__ void hgrn_ternary_mm_f32(
        const float * x, const uint8_t * wq, const float * scale, float * dst,
        const int64_t in_dim, const int64_t out_dim, const int64_t n_tok, const int64_t row_bytes,
        int act_quant, int frac_bits) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= out_dim * n_tok) {
        return;
    }

    const int64_t o = gid % out_dim;
    const int64_t t = gid / out_dim;
    const int64_t n_blocks = in_dim / 256;

    const float   * xrow    = x  + t * in_dim;
    const uint8_t * wpacked = wq + o * row_bytes;

    int8_t wblk[256];

    if (act_quant == GGML_HGRN_ACT_QUANT_FIXEDQ510) {
        const float qs        = (float) (1 << frac_bits);
        const float inv_fixed = 1.0f / (qs * scale[0]);

        int32_t acc = 0;
        for (int64_t b = 0; b < n_blocks; b++) {
            hgrn_tq1_unpack_block(wblk, wpacked + b * 52);
            const float * xblk = xrow + b * 256;
            for (int64_t k = 0; k < 256; k++) {
                float q = nearbyintf(xblk[k] * qs);
                if (q > 32767.0f) q = 32767.0f;
                else if (q < -32768.0f) q = -32768.0f;
                const int32_t yq = (int32_t) q;
                const int8_t  wk = wblk[k];
                if (wk > 0) acc += yq;
                else if (wk < 0) acc -= yq;
            }
        }
        dst[t * out_dim + o] = (float) acc * inv_fixed;
        return;
    }

    float acc = 0.0f;
    for (int64_t b = 0; b < n_blocks; b++) {
        hgrn_tq1_unpack_block(wblk, wpacked + b * 52);
        const float * xblk = xrow + b * 256;
        for (int64_t k = 0; k < 256; k++) {
            acc += xblk[k] * (float) wblk[k];
        }
    }

    dst[t * out_dim + o] = acc / scale[0];
}

// HGRN-Bit gated linear recurrence: h_t = f_t*h_{t-1} + i_t, y_t = h_t
// state is purely elementwise (no cross-dim mixing), so one thread scans each (d, h, s)
// independently over all T steps; dst = [ y (D*H*T*S) ; new state (D*H*S) ]
static __global__ void hgrn_scan_f32(
        const float * ii, const float * ff, const float * state_in, float * dst,
        const int64_t D, const int64_t H, const int64_t T, const int64_t S) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= D * H * S) {
        return;
    }

    const int64_t d = gid % D;
    const int64_t h = (gid / D) % H;
    const int64_t s = gid / (D * H);

    float state = state_in[(s * H + h) * D + d];

    float * yout = dst;
    float * sout = dst + D * H * T * S;

    for (int64_t t = 0; t < T; t++) {
        const int64_t off = ((s * T + t) * H + h) * D + d;
        state = ff[off] * state + ii[off];
        yout[off] = state;
    }

    sout[(s * H + h) * D + d] = state;
}

void ggml_cuda_op_hgrn_ternary_mm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x  = dst->src[0];
    const ggml_tensor * wq = dst->src[1];
    const ggml_tensor * sc = dst->src[2];

    GGML_ASSERT(x->type  == GGML_TYPE_F32);
    GGML_ASSERT(wq->type == GGML_TYPE_I8);
    GGML_ASSERT(sc->type == GGML_TYPE_F32);

    const int32_t act_quant = ggml_get_op_params_i32(dst, 0);
    const int32_t frac_bits = ggml_get_op_params_i32(dst, 1);

    const int64_t in_dim    = x->ne[0];
    const int64_t out_dim   = wq->ne[1];
    const int64_t n_tok     = x->ne[1] * x->ne[2] * x->ne[3];
    const int64_t row_bytes = wq->ne[0];  // TQ1_0-style packed row: in_dim/256*52

    const float   * x_d  = (const float   *) x->data;
    const uint8_t * wq_d = (const uint8_t *) wq->data;
    const float   * sc_d = (const float   *) sc->data;
    float         * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t n_threads_total = out_dim * n_tok;
    const int block_size = 256;
    const int64_t n_blocks = (n_threads_total + block_size - 1) / block_size;

    hgrn_ternary_mm_f32<<<n_blocks, block_size, 0, stream>>>(x_d, wq_d, sc_d, dst_d, in_dim, out_dim, n_tok, row_bytes, act_quant, frac_bits);
}

void ggml_cuda_op_hgrn_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ti = dst->src[0];
    const ggml_tensor * tf = dst->src[1];
    const ggml_tensor * ts = dst->src[2];

    GGML_ASSERT(ti->type == GGML_TYPE_F32);
    GGML_ASSERT(tf->type == GGML_TYPE_F32);
    GGML_ASSERT(ts->type == GGML_TYPE_F32);

    const int64_t D = ti->ne[0];
    const int64_t H = ti->ne[1];
    const int64_t T = ti->ne[2];
    const int64_t S = ti->ne[3];

    const float * i_d = (const float *) ti->data;
    const float * f_d = (const float *) tf->data;
    const float * s_d = (const float *) ts->data;
    float        * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t n_threads_total = D * H * S;
    const int block_size = 256;
    const int64_t n_blocks = (n_threads_total + block_size - 1) / block_size;

    hgrn_scan_f32<<<n_blocks, block_size, 0, stream>>>(i_d, f_d, s_d, dst_d, D, H, T, S);
}
