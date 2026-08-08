#include "common.cuh"
#include "hgrn.cuh"

// HGRN-Bit (MatMul-Free LM) ternary BitLinear: y = (x_norm @ wq) / scale_w, wq in {-1,0,+1}
// one thread per (o, t) output element, full serial dot product over in_dim (correctness-first,
// same math as the CPU/Metal ports - see src/models/hgrnbit.cpp)
static __global__ void hgrn_ternary_mm_f32(
        const float * x, const int8_t * wq, const float * scale, float * dst,
        const int64_t in_dim, const int64_t out_dim, const int64_t n_tok) {
    const int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= out_dim * n_tok) {
        return;
    }

    const int64_t o = gid % out_dim;
    const int64_t t = gid / out_dim;

    const float  * xrow = x  + t * in_dim;
    const int8_t * wrow = wq + o * in_dim;

    float acc = 0.0f;
    for (int64_t k = 0; k < in_dim; k++) {
        acc += xrow[k] * (float) wrow[k];
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

    const int64_t in_dim  = x->ne[0];
    const int64_t out_dim = wq->ne[1];
    const int64_t n_tok   = x->ne[1] * x->ne[2] * x->ne[3];

    const float  * x_d  = (const float  *) x->data;
    const int8_t * wq_d = (const int8_t *) wq->data;
    const float  * sc_d = (const float  *) sc->data;
    float        * dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t n_threads_total = out_dim * n_tok;
    const int block_size = 256;
    const int64_t n_blocks = (n_threads_total + block_size - 1) / block_size;

    hgrn_ternary_mm_f32<<<n_blocks, block_size, 0, stream>>>(x_d, wq_d, sc_d, dst_d, in_dim, out_dim, n_tok);
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
