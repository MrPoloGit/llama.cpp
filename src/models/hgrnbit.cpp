#include "models.h"

#include "llama-memory-recurrent.h"

#include <vector>

// Custom ops for HGRN-Bit (MatMul-Free LM). ggml has no native op for either of these, so
// they run as plain CPU callbacks via ggml_custom_4d instead of new GGML_OP_* types - see
// matmulfreellmCPU/cpp/src/kernels/{bitlinear,hgrn_scan}.cpp for the reference this is ported
// from (ActQuant::Float mode: no activation quantization, matches the published HF model).

// y = (x_norm @ wq) / scale_w, wq in {-1,0,+1}. src[0]=x_norm [in,T] f32, src[1]=wq [in,out] i8,
// src[2]=scale_w [1] f32. dst = [out,T] f32.
static void hgrn_ternary_matmul_cb(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

    const struct ggml_tensor * x  = dst->src[0];
    const struct ggml_tensor * wq = dst->src[1];
    const struct ggml_tensor * sc = dst->src[2];

    const int64_t in_dim  = x->ne[0];
    const int64_t n_tok   = x->ne[1];
    const int64_t out_dim = wq->ne[1];

    const float  * xd = (const float  *) x->data;
    const int8_t * wd = (const int8_t *) wq->data;
    const float scale_w = *(const float *) sc->data;

    float * yd = (float *) dst->data;

    for (int64_t o = ith; o < out_dim; o += nth) {
        const int8_t * wrow = wd + o * in_dim;
        for (int64_t t = 0; t < n_tok; ++t) {
            const float * xrow = xd + t * in_dim;
            float acc = 0.0f;
            for (int64_t k = 0; k < in_dim; ++k) {
                acc += xrow[k] * (float) wrow[k];
            }
            yd[t * out_dim + o] = acc / scale_w;
        }
    }
}

// gated linear recurrence: h_t = f_t*h_{t-1} + i_t, y_t = h_t, scanned per (head, seq) stream.
// src[0]=i [D,H,T,S] f32, src[1]=f [D,H,T,S] f32, src[2]=state_in [D,H,S] f32.
// dst = 1D f32, [ y (D*H*T*S) ; state_out (D*H*S) ] concatenated, split by the caller.
static void hgrn_scan_cb(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(userdata);

    const struct ggml_tensor * ti = dst->src[0];
    const struct ggml_tensor * tf = dst->src[1];
    const struct ggml_tensor * ts = dst->src[2];

    const int64_t D = ti->ne[0];
    const int64_t H = ti->ne[1];
    const int64_t T = ti->ne[2];
    const int64_t S = ti->ne[3];

    const float * id = (const float *) ti->data;
    const float * fd = (const float *) tf->data;
    const float * s0 = (const float *) ts->data;

    float * yout = (float *) dst->data;
    float * sout = yout + D * H * T * S;

    std::vector<float> state(D);

    const int64_t n_streams = H * S;
    for (int64_t st = ith; st < n_streams; st += nth) {
        const int64_t h = st % H;
        const int64_t s = st / H;

        const float * state_init = s0 + (s * H + h) * D;
        std::copy(state_init, state_init + D, state.begin());

        for (int64_t t = 0; t < T; ++t) {
            const float * irow = id + ((s * T + t) * H + h) * D;
            const float * frow = fd + ((s * T + t) * H + h) * D;
            float       * yrow = yout + ((s * T + t) * H + h) * D;
            for (int64_t d = 0; d < D; ++d) {
                state[d] = frow[d] * state[d] + irow[d];
                yrow[d]  = state[d];
            }
        }

        float * srow = sout + (s * H + h) * D;
        std::copy(state.begin(), state.end(), srow);
    }
}

void llama_model_hgrnbit::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_HGRNBIT_HEAD_DIM,             hparams.hgrnbit_head_dim);
}

void llama_model_hgrnbit::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t input_dim = (int64_t) hparams.n_head() * hparams.hgrnbit_head_dim;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    output          = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);
    output_bitnorm  = create_tensor(tn(LLM_TENSOR_OUTPUT, "norm"),   {n_embd}, 0);
    output_bitscale = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"),  {1}, 0);

    hgrn_lower_bounds = create_tensor(tn(LLM_TENSOR_HGRN_LOWER_BOUNDS, "weight"), {n_embd, n_layer}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);
        layer.hgrn_g_norm = create_tensor(tn(LLM_TENSOR_HGRN_GNORM, "weight", i), {input_dim}, 0);

        layer.hgrn_iproj       = create_tensor(tn(LLM_TENSOR_HGRN_IPROJ, "weight", i), {n_embd, input_dim}, 0);
        layer.hgrn_iproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_IPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_iproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_IPROJ, "scale",  i), {1}, 0);

        layer.hgrn_fproj       = create_tensor(tn(LLM_TENSOR_HGRN_FPROJ, "weight", i), {n_embd, input_dim}, 0);
        layer.hgrn_fproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_FPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_fproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_FPROJ, "scale",  i), {1}, 0);

        layer.hgrn_gproj       = create_tensor(tn(LLM_TENSOR_HGRN_GPROJ, "weight", i), {n_embd, input_dim}, 0);
        layer.hgrn_gproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_GPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_gproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_GPROJ, "scale",  i), {1}, 0);

        layer.hgrn_oproj       = create_tensor(tn(LLM_TENSOR_HGRN_OPROJ, "weight", i), {input_dim, n_embd}, 0);
        layer.hgrn_oproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_OPROJ, "norm",   i), {input_dim}, 0);
        layer.hgrn_oproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_OPROJ, "scale",  i), {1}, 0);

        layer.hgrn_gateproj       = create_tensor(tn(LLM_TENSOR_HGRN_GATEPROJ, "weight", i), {n_embd, 2 * n_ff}, 0);
        layer.hgrn_gateproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_GATEPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_gateproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_GATEPROJ, "scale",  i), {1}, 0);

        layer.hgrn_downproj       = create_tensor(tn(LLM_TENSOR_HGRN_DOWNPROJ, "weight", i), {n_ff, n_embd}, 0);
        layer.hgrn_downproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_DOWNPROJ, "norm",   i), {n_ff}, 0);
        layer.hgrn_downproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_DOWNPROJ, "scale",  i), {1}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_hgrnbit::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

ggml_tensor * llama_model_hgrnbit::graph::build_hgrn_bitlinear(
        ggml_tensor * cur,
        ggml_tensor * norm,
        ggml_tensor * wq,
        ggml_tensor * scale,
        int il) const {
    cur = build_norm(cur, norm, nullptr, LLM_NORM_RMS, il);
    cur = ggml_cont(ctx0, cur);

    ggml_tensor * args[3] = { cur, wq, scale };

    ggml_tensor * y = ggml_custom_4d(ctx0, GGML_TYPE_F32, wq->ne[1], cur->ne[1], cur->ne[2], cur->ne[3],
                                      args, 3, hgrn_ternary_matmul_cb, GGML_N_TASKS_MAX, nullptr);
    cb(y, "hgrn_bitlinear", il);

    return y;
}

ggml_tensor * llama_model_hgrnbit::graph::build_hgrn_scan(
        llm_graph_input_rs * inp,
        ggml_tensor * i,
        ggml_tensor * f,
        int il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_seqs       = ubatch.n_seqs;

    const int64_t head_dim = hparams.hgrnbit_head_dim;
    const int64_t n_head_l = hparams.n_head(il);

    const auto kv_head = mctx_cur->get_head();

    ggml_tensor * state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

    ggml_tensor * i4 = ggml_reshape_4d(ctx0, ggml_cont(ctx0, i), head_dim, n_head_l, n_seq_tokens, n_seqs);
    ggml_tensor * f4 = ggml_reshape_4d(ctx0, ggml_cont(ctx0, f), head_dim, n_head_l, n_seq_tokens, n_seqs);
    ggml_tensor * s3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, state), head_dim, n_head_l, n_seqs);

    const int64_t n_y = head_dim * n_head_l * n_seq_tokens * n_seqs;
    const int64_t n_s = head_dim * n_head_l * n_seqs;

    ggml_tensor * args[3] = { i4, f4, s3 };
    ggml_tensor * scan_out = ggml_custom_4d(ctx0, GGML_TYPE_F32, n_y + n_s, 1, 1, 1,
                                             args, 3, hgrn_scan_cb, GGML_N_TASKS_MAX, nullptr);
    cb(scan_out, "hgrn_scan", il);

    ggml_tensor * y         = ggml_view_1d(ctx0, scan_out, n_y, 0);
    ggml_tensor * state_out = ggml_view_1d(ctx0, scan_out, n_s, n_y * ggml_element_size(scan_out));

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, state_out,
        ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                     hparams.n_embd_s() * kv_head * ggml_element_size(mctx_cur->get_s_l(il)))));

    return ggml_reshape_2d(ctx0, y, head_dim * n_head_l, n_seq_tokens * n_seqs);
}

llama_model_hgrnbit::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const llama_layer & layer = model.layers[il];

        ggml_tensor * resid = inpL;

        ggml_tensor * hs = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(hs, "attn_norm", il);

        ggml_tensor * i = build_hgrn_bitlinear(hs, layer.hgrn_iproj_norm, layer.hgrn_iproj, layer.hgrn_iproj_scale, il);
        ggml_tensor * f = build_hgrn_bitlinear(hs, layer.hgrn_fproj_norm, layer.hgrn_fproj, layer.hgrn_fproj_scale, il);
        ggml_tensor * g = build_hgrn_bitlinear(hs, layer.hgrn_gproj_norm, layer.hgrn_gproj, layer.hgrn_gproj_scale, il);

        f = ggml_sigmoid(ctx0, f);

        if (model.hgrn_lower_bounds && il > 0) {
            ggml_tensor * lb = ggml_view_1d(ctx0, model.hgrn_lower_bounds, n_embd,
                                             il * n_embd * ggml_element_size(model.hgrn_lower_bounds));
            // f = lb + (1 - lb) * f = f - lb*f + lb
            f = ggml_add(ctx0, ggml_sub(ctx0, f, ggml_mul(ctx0, f, lb)), lb);
        }

        ggml_tensor * one_minus_f = ggml_scale_bias(ctx0, f, -1.0f, 1.0f);
        i = ggml_mul(ctx0, ggml_silu(ctx0, i), one_minus_f);

        ggml_tensor * recur = build_hgrn_scan(rs_inp, i, f, il);

        ggml_tensor * g_normed = build_norm(g, layer.hgrn_g_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * oin      = ggml_mul(ctx0, g_normed, ggml_silu(ctx0, recur));

        ggml_tensor * oout = build_hgrn_bitlinear(oin, layer.hgrn_oproj_norm, layer.hgrn_oproj, layer.hgrn_oproj_scale, il);

        cur = ggml_add(ctx0, resid, oout);
        cb(cur, "attn_out", il);

        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        resid = cur;

        ggml_tensor * hs2 = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * y = build_hgrn_bitlinear(hs2, layer.hgrn_gateproj_norm, layer.hgrn_gateproj, layer.hgrn_gateproj_scale, il);

        const int64_t n_ff_cur = y->ne[0] / 2;
        const int64_t n_tok    = y->ne[1];
        ggml_tensor * gate = ggml_cont(ctx0, ggml_view_2d(ctx0, y, n_ff_cur, n_tok, y->nb[1], 0));
        ggml_tensor * up   = ggml_cont(ctx0, ggml_view_2d(ctx0, y, n_ff_cur, n_tok, y->nb[1], n_ff_cur * ggml_element_size(y)));

        ggml_tensor * z = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
        ggml_tensor * zout = build_hgrn_bitlinear(z, layer.hgrn_downproj_norm, layer.hgrn_downproj, layer.hgrn_downproj_scale, il);

        cur = ggml_add(ctx0, resid, zout);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_hgrn_bitlinear(cur, model.output_bitnorm, model.output, model.output_bitscale, -1);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
