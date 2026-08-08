from __future__ import annotations

import numpy as np

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf

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

    def _add_bitlinear(self, base: str, weight: Tensor):
        # ref: matmulfreellmCPU/tools/reference.py weight_quant_int
        w = weight.float()
        scale_w = w.abs().mean().clamp(min=1e-5).reciprocal()
        wq = (w * scale_w).round().clamp(-1, 1).numpy().astype(np.int8)
        self.gguf_writer.add_tensor(f"{base}.weight", wq, raw_dtype=gguf.GGMLQuantizationType.I8)
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
