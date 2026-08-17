#include "models.h"

#include "llama-memory-recurrent.h"

// HGRN-Bit (MatMul-Free LM) ternary BitLinear + gated recurrence are ggml_hgrn_ternary_mm /
// ggml_hgrn_scan (ggml.h), first-class ops with CPU (ggml-cpu/ops.cpp), Metal
// (ggml-metal.metal), and CUDA (ggml-cuda/hgrn.cu) implementations, ported from
// matmulfreellmCPU/cpp/src/kernels/{bitlinear,hgrn_scan}.cpp. Float activation quant
// (no activation quantization) matches the published HF checkpoints and is the default;
// FixedQ510 (matmulfreellmCPU's ActQuant::FixedQ510, its default and the actual FPGA
// datapath) is CPU-only so far - see LLM_KV_HGRNBIT_ACT_QUANT_MODE and
// ggml_compute_forward_hgrn_ternary_mm's FixedQ510 branch. BitLinear weight rows are
// TQ1_0-style packed (5 trits/byte, see ggml_hgrn_ternary_mm), ported from
// matmulfreellmCPU/cpp/include/mmfree/tq1.hpp.

void llama_model_hgrnbit::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_HGRNBIT_HEAD_DIM,             hparams.hgrnbit_head_dim);
    // Both optional: absent in every GGUF converted before FixedQ510 support (and in every
    // GGUF converted without conversion/hgrnbit.py's HGRNBIT_ACT_QUANT=fixedq510 env var),
    // in which case the hparams.h defaults (Float, frac_bits=10) preserve prior behavior.
    ml.get_key(LLM_KV_HGRNBIT_ACT_QUANT_MODE, hparams.hgrnbit_act_quant_mode, false);
    ml.get_key(LLM_KV_HGRNBIT_FRAC_BITS,      hparams.hgrnbit_frac_bits,      false);
}

// BitLinear weight rows are packed TQ1_0-style (see ggml_hgrn_ternary_mm): 256 ternary
// weights -> 52 bytes. in_dim must be a multiple of 256 (true for every dim in the published
// HGRN-Bit checkpoints - hidden_size and the 256-rounded intermediate_size both qualify).
static int64_t hgrn_tq1_row_bytes(int64_t in_dim) {
    GGML_ASSERT(in_dim % 256 == 0);
    return (in_dim / 256) * 52;
}

void llama_model_hgrnbit::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t input_dim = (int64_t) hparams.n_head() * hparams.hgrnbit_head_dim;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    output          = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {hgrn_tq1_row_bytes(n_embd), n_vocab}, 0);
    output_bitnorm  = create_tensor(tn(LLM_TENSOR_OUTPUT, "norm"),   {n_embd}, 0);
    output_bitscale = create_tensor(tn(LLM_TENSOR_OUTPUT, "scale"),  {1}, 0);

    hgrn_lower_bounds = create_tensor(tn(LLM_TENSOR_HGRN_LOWER_BOUNDS, "weight"), {n_embd, n_layer}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);
        layer.hgrn_g_norm = create_tensor(tn(LLM_TENSOR_HGRN_GNORM, "weight", i), {input_dim}, 0);

        layer.hgrn_iproj       = create_tensor(tn(LLM_TENSOR_HGRN_IPROJ, "weight", i), {hgrn_tq1_row_bytes(n_embd), input_dim}, 0);
        layer.hgrn_iproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_IPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_iproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_IPROJ, "scale",  i), {1}, 0);

        layer.hgrn_fproj       = create_tensor(tn(LLM_TENSOR_HGRN_FPROJ, "weight", i), {hgrn_tq1_row_bytes(n_embd), input_dim}, 0);
        layer.hgrn_fproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_FPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_fproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_FPROJ, "scale",  i), {1}, 0);

        layer.hgrn_gproj       = create_tensor(tn(LLM_TENSOR_HGRN_GPROJ, "weight", i), {hgrn_tq1_row_bytes(n_embd), input_dim}, 0);
        layer.hgrn_gproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_GPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_gproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_GPROJ, "scale",  i), {1}, 0);

        layer.hgrn_oproj       = create_tensor(tn(LLM_TENSOR_HGRN_OPROJ, "weight", i), {hgrn_tq1_row_bytes(input_dim), n_embd}, 0);
        layer.hgrn_oproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_OPROJ, "norm",   i), {input_dim}, 0);
        layer.hgrn_oproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_OPROJ, "scale",  i), {1}, 0);

        layer.hgrn_gateproj       = create_tensor(tn(LLM_TENSOR_HGRN_GATEPROJ, "weight", i), {hgrn_tq1_row_bytes(n_embd), 2 * n_ff}, 0);
        layer.hgrn_gateproj_norm  = create_tensor(tn(LLM_TENSOR_HGRN_GATEPROJ, "norm",   i), {n_embd}, 0);
        layer.hgrn_gateproj_scale = create_tensor(tn(LLM_TENSOR_HGRN_GATEPROJ, "scale",  i), {1}, 0);

        layer.hgrn_downproj       = create_tensor(tn(LLM_TENSOR_HGRN_DOWNPROJ, "weight", i), {hgrn_tq1_row_bytes(n_ff), n_embd}, 0);
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

    const auto act_quant = (ggml_hgrn_act_quant) hparams.hgrnbit_act_quant_mode;
    ggml_tensor * y = ggml_hgrn_ternary_mm(ctx0, cur, wq, scale, act_quant, hparams.hgrnbit_frac_bits);
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

    ggml_tensor * scan_out = ggml_hgrn_scan(ctx0, i4, f4, s3);
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
