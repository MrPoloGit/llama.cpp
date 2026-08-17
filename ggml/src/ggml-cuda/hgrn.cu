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
// (same math as the CPU/Metal ports - see src/models/hgrnbit.cpp)
// One warp (32 threads) per (o, t) output element: each lane strides over a subset of the
// 256-element blocks with its own independent accumulator (breaking the long serial FMA
// chain a single-thread reduction would have - see matmulfreellmCPU's simd.hpp comment on
// exactly this problem for the same-shape workload), then warp_reduce_sum (common.cuh)
// combines the 32 partials. Mirrors the simdgroup/simd_sum fix already applied to the Metal
// kernel (see ggml-metal.metal) - same root cause, same fix, CUDA's warp-shuffle idiom
// instead of Metal's simd_sum. 32 lanes covers every projection's block count in every
// published checkpoint (max 27, at 2.7B's down_proj) with only marginal idle-lane waste
// beyond that. warp_reduce_sum (not a hand-rolled __shfl_down_sync loop) so this stays
// correct on the HIP/ROCm and MUSA builds that also compile this file.
static __global__ void hgrn_ternary_mm_f32(
        const float * x, const uint8_t * wq, const float * scale, float * dst,
        const int64_t in_dim, const int64_t out_dim, const int64_t n_tok, const int64_t row_bytes) {
    const int64_t gid  = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int     lane = threadIdx.x % 32;
    if (gid >= out_dim * n_tok) {
        return;
    }

    const int64_t o = gid % out_dim;
    const int64_t t = gid / out_dim;
    const int64_t n_blocks = in_dim / 256;

    const float   * xrow    = x  + t * in_dim;
    const uint8_t * wpacked = wq + o * row_bytes;

    float acc = 0.0f;
    int8_t wblk[256];
    for (int64_t b = lane; b < n_blocks; b += 32) {
        hgrn_tq1_unpack_block(wblk, wpacked + b * 52);
        const float * xblk = xrow + b * 256;
        for (int64_t k = 0; k < 256; k++) {
            acc += xblk[k] * (float) wblk[k];
        }
    }

    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        dst[t * out_dim + o] = acc / scale[0];
    }
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

    const int64_t in_dim    = x->ne[0];
    const int64_t out_dim   = wq->ne[1];
    const int64_t n_tok     = x->ne[1] * x->ne[2] * x->ne[3];
    const int64_t row_bytes = wq->ne[0];  // TQ1_0-style packed row: in_dim/256*52

    const float   * x_d  = (const float   *) x->data;
    const uint8_t * wq_d = (const uint8_t *) wq->data;
    const float   * sc_d = (const float   *) sc->data;
    float         * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    // one warp (32 threads) per output element - see the kernel comment for why
    const int64_t n_threads_total = out_dim * n_tok * 32;
    const int block_size = 128; // 4 warps/block, must stay a multiple of 32
    const int64_t n_blocks = (n_threads_total + block_size - 1) / block_size;

    hgrn_ternary_mm_f32<<<n_blocks, block_size, 0, stream>>>(x_d, wq_d, sc_d, dst_d, in_dim, out_dim, n_tok, row_bytes);
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
