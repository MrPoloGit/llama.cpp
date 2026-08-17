from __future__ import annotations

import os

import numpy as np

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf

# (blk tensor name tag, HF weight infix under "model.layers.{bid}.") for the six
# ternary BitLinear projections every HGRN-Bit layer has.
LAYER_PROJS = [
    ("hgrn_iproj", "attn.i_proj"),
    ("hgrn_fproj", "attn.f_proj"),
    ("hgrn_gproj", "attn.g_proj"),
    ("hgrn_oproj", "attn.o_proj"),
    ("hgrn_gateproj", "mlp.gate_proj"),
    ("hgrn_downproj", "mlp.down_proj"),
]


def _pack_tq1(wq: np.ndarray) -> np.ndarray:
    # Pack ternary weights ({-1,0,+1}, int8) into ggml-quants' TQ1_0 block layout minus its
    # per-block fp16 scale (HGRN-Bit already has one scale per whole projection): 256 ternary
    # weights -> 52 bytes/block (48 "qs" + 4 "qh"), 5 trits/byte via base-3 packing.
    # ref: matmulfreellmCPU/cpp/include/mmfree/tq1.hpp pack_row() - same encode, vectorized.
    #   group A: qs[0..31],  5 trits each -> elements j + 32*l   (j<32, l<5) = elements 0..159
    #   group B: qs[32..47], 5 trits each -> elements 160+j+16*l (j<16, l<5) = elements 160..239
    #   group C: qh[0..3],   4 trits each -> elements 240+j+4*l  (j<4,  l<4) = elements 240..255
    #   v = sum_l u_l * 3^(len-1-l), u_l = trit_l + 1 in {0,1,2}; q = ceil(v * 256 / 3^len)
    rows, n = wq.shape
    assert n % 256 == 0, f"in_dim={n} must be a multiple of 256 for TQ1_0 packing"
    nb = n // 256
    w = wq.reshape(rows, nb, 256).astype(np.int64) + 1  # trit -> u in {0,1,2}

    pow_a = np.array([81, 27, 9, 3, 1], dtype=np.int64).reshape(1, 1, 5, 1)
    pow_c = np.array([27, 9, 3, 1], dtype=np.int64).reshape(1, 1, 4, 1)

    v_a = (w[:, :, :160].reshape(rows, nb, 5, 32) * pow_a).sum(axis=2)
    v_b = (w[:, :, 160:240].reshape(rows, nb, 5, 16) * pow_a).sum(axis=2)
    v_c = (w[:, :, 240:256].reshape(rows, nb, 4, 4) * pow_c).sum(axis=2)

    qs_a = ((v_a * 256 + 242) // 243).astype(np.uint8)
    qs_b = ((v_b * 256 + 242) // 243).astype(np.uint8)
    qh   = ((v_c * 256 + 80) // 81).astype(np.uint8)

    packed = np.concatenate([qs_a, qs_b, qh], axis=2)  # [rows, nb, 52]
    return packed.reshape(rows, nb * 52).view(np.int8)


@ModelBase.register("HGRNBitForCausalLM")
class HgrnBitModel(TextModel):
    model_arch = gguf.MODEL_ARCH.HGRNBIT

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        hidden_size = self.hparams["hidden_size"]
        expand_ratio = self.hparams.get("expand_ratio", 1)
        num_heads = self.hparams.get("num_heads", 1)
        hidden_ratio = self.hparams.get("hidden_ratio", 4)
        input_dim = hidden_size * expand_ratio

        intermediate_size = self.hparams.get("intermediate_size")
        if intermediate_size is None:
            # ref: matmulfreellmCPU/tools/reference.py HGRNBitConfig.inter_size
            i = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((i + 255) // 256)

        # HGRN-Bit is a linear recurrence, not attention-window limited.
        self.gguf_writer.add_context_length(1048576)
        self.gguf_writer.add_embedding_length(hidden_size)
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_feed_forward_length(intermediate_size)
        self.gguf_writer.add_head_count(num_heads)
        self.gguf_writer.add_hgrnbit_head_dim(input_dim // num_heads)
        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("rms_norm_eps", 1e-6))
        self.gguf_writer.add_file_type(self.ftype)

        # Activation quant for the BitLinear projections - mirrors matmulfreellmCPU's
        # ActQuant enum (mmfree/kernels.hpp). Defaults to float (the published-checkpoint
        # "triton" golden, matching this converter's historical behavior) unless overridden;
        # opt into FixedQ510 (matmulfreellmCPU's own default, its FPGA-datapath numerics) via
        # env vars, e.g. `HGRNBIT_ACT_QUANT=fixedq510 python3 convert_hf_to_gguf.py ...`.
        # No CLI flag: this is a niche, single-architecture option, not worth adding to the
        # shared convert_hf_to_gguf.py argparse surface every architecture goes through.
        act_quant = os.environ.get("HGRNBIT_ACT_QUANT", "float").strip().lower()
        if act_quant not in ("float", "fixedq510"):
            raise ValueError(f"HGRNBIT_ACT_QUANT must be 'float' or 'fixedq510', got {act_quant!r}")
        if act_quant == "fixedq510":
            frac_bits = int(os.environ.get("HGRNBIT_FRAC_BITS", "10"))
            self.gguf_writer.add_hgrnbit_act_quant_mode(1)  # GGML_HGRN_ACT_QUANT_FIXEDQ510
            self.gguf_writer.add_hgrnbit_frac_bits(frac_bits)
        # act_quant == "float": write nothing: the loader defaults (Float, frac_bits=10)
        # match this converter's prior, still-unconditional behavior for every existing GGUF.

    def _add_bitlinear(self, base: str, weight: Tensor):
        # ref: matmulfreellmCPU/tools/reference.py weight_quant_int
        # Force real materialization before _pack_tq1: it uses plain numpy free functions
        # (np.concatenate) that gguf-py's LazyNumpyTensor doesn't intercept (no
        # __array_function__ yet - see gguf-py/gguf/lazy.py), so calling them on a
        # still-lazy tensor silently packs the zero-filled meta placeholder instead of
        # real weights.
        w = LazyTorchTensor.to_eager(weight).float()
        scale_w = w.abs().mean().clamp(min=1e-5).reciprocal()
        wq = (w * scale_w).round().clamp(-1, 1).numpy().astype(np.int8)
        packed = _pack_tq1(wq)
        self.gguf_writer.add_tensor(f"{base}.weight", packed, raw_dtype=gguf.GGMLQuantizationType.I8)
        self.gguf_writer.add_tensor(f"{base}.scale", scale_w.reshape(1).numpy().astype(np.float32))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "model.lower_bounds":
            lb = data_torch.float().softmax(0)
            lb = lb.cumsum(0) - lb[0]
            self.gguf_writer.add_tensor("hgrn_lower_bounds.weight", lb.numpy().astype(np.float32))
            return

        if name == "lm_head.weight":
            self._add_bitlinear("output", data_torch)
            return
        if name == "lm_head.norm.weight":
            self.gguf_writer.add_tensor("output.norm", data_torch.float().numpy().astype(np.float32))
            return

        if bid is not None:
            for tag, infix in LAYER_PROJS:
                if name == f"model.layers.{bid}.{infix}.weight":
                    self._add_bitlinear(f"blk.{bid}.{tag}", data_torch)
                    return
                if name == f"model.layers.{bid}.{infix}.norm.weight":
                    self.gguf_writer.add_tensor(f"blk.{bid}.{tag}.norm", data_torch.float().numpy().astype(np.float32))
                    return
            if name == f"model.layers.{bid}.attn.g_norm.weight":
                self.gguf_writer.add_tensor(f"blk.{bid}.hgrn_g_norm.weight", data_torch.float().numpy().astype(np.float32))
                return

        yield from super().modify_tensors(data_torch, name, bid)
