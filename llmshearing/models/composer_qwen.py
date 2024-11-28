import math
from einops import rearrange
import torch
import torch.nn as nn
from llmshearing.models.composer_llama import (
    ComposerMosaicLlama,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    normal_attn_fn,
    flash_attn_fn,
    prepare_decoder_attention_mask,
    turn_mlp_z,
)
from omegaconf import DictConfig
from typing import List, Optional, Tuple
from llmshearing.models.l0_module import L0Module
from transformers.pytorch_utils import (
    prune_linear_layer,
)
import torch.nn.functional as F


class ComposerMosaicQwen(ComposerMosaicLlama):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = QwenModel(cfg)


class QwenModel(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        print(f"Tried to build Qwen model with cfg.name={cfg.name}")
        self.cfg = cfg

        ### added ###
        self.l0_module = None
        if getattr(self.cfg, "l0_module", None) is not None:
            self.l0_module = L0Module(self.cfg, device=cfg.init_device)

        self.attn_impl = cfg.attn_impl

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    cfg.vocab_size, cfg.d_model, device=cfg.init_device
                ),
            }
        )

        self.transformer.update(
            {
                "blocks": nn.ModuleList(
                    [
                        QwenBlock(cfg, device=cfg.init_device)
                        for _ in range(cfg.n_layers)
                    ]
                )
            }
        )

        self.transformer.update(
            {
                "output": nn.Linear(
                    cfg.d_model, cfg.vocab_size, device=cfg.init_device, bias=False
                )
            }
        )

        # Tie weights
        self.transformer.wte.weight = self.transformer.output.weight

        self.transformer.update(
            {
                "ln_f": LlamaRMSNorm(
                    cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=cfg.init_device
                )
            }
        )

        self.is_causal = True
        if cfg.get("verbose", 0) > 2:
            print(self)

    def prune_params(self, zs=None):
        if zs is None:
            self.l0_module.eval()
            zs = self.l0_module(calculate_lagrangian=False)

        if "hidden_z" in zs:
            hidden_z = zs["hidden_z"]
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            self.transformer.ln_f.prune_params(hidden_z)
            self.transformer.wte.weight.data = self.transformer.wte.weight.data.mul(
                hidden_z
            )
            self.transformer.wte.weight = torch.nn.parameter.Parameter(
                self.transformer.wte.weight.index_select(1, remaining_index).clone()
            )
            self.transformer.wte.embedding_dim = len(remaining_index)
            # self.transformer.output.weight.data = self.transformer.output.weight.data.mul(hidden_z)
            half = self.transformer.output.weight.data.dtype == torch.float16
            self.transformer.output = prune_linear_layer(
                self.transformer.output, remaining_index, dim=1
            )
            if half:
                self.transformer.output = self.transformer.output.half()

        for i, block in enumerate(self.transformer.blocks):
            zs_block = self.get_zs_block(zs, i)
            block.prune_params(zs_block)

    def get_zs_block(self, zs, block_idx):
        zs_block = {}
        if zs is not None:
            for key in zs:
                if key == "hidden_z":
                    zs_block["hidden_z"] = zs["hidden_z"]
                else:
                    zs_block[key] = zs[key][block_idx]
        return zs_block

    def forward(
        self,
        input_ids: torch.LongTensor,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        pruned_steps: int = 0,
        retain_grad: bool = False,
        **zs,
    ):
        S = input_ids.size(1)
        assert (
            S <= self.cfg.max_seq_len
        ), f"Sequence length ({S}) exceeds model maximum sequence length ({self.cfg.max_seq_len})!"

        tok_emb = self.transformer.wte(input_ids)
        if "hidden_z" in zs:
            tok_emb = tok_emb.mul(zs["hidden_z"])

        x = tok_emb

        attn_bias = None  # only consider the flash attention case
        attention_mask = prepare_decoder_attention_mask(
            (tok_emb.size(0), tok_emb.size(1)), tok_emb
        )

        l0_output = None
        if self.l0_module is not None:
            assert zs == {}, "zs should be empty when using L0Module"
            zs = self.l0_module(calculate_lagrangian=False, pruned_steps=pruned_steps)

        for b_idx, block in enumerate(self.transformer.blocks):
            zs_block = self.get_zs_block(zs, b_idx)
            past_key_value = (
                past_key_values[b_idx] if past_key_values is not None else None
            )

            x, past_key_value = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                key_padding_mask=key_padding_mask,
                is_causal=self.is_causal,
                attention_mask=attention_mask,
                retain_grad=retain_grad,
                **zs_block,
            )

            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value

        x = self.transformer.ln_f(x, hidden_z=zs.get("hidden_z", None))
        logits = self.transformer.output(x)

        if self.l0_module is not None:
            l0_output = self.l0_module(
                calculate_lagrangian=True, pruned_steps=pruned_steps
            )

        return {"logits": logits, "l0_output": l0_output, "zs": zs}

    def param_init_fn(self, module):
        pass

    def fsdp_wrap_fn(self, module):
        return isinstance(module, QwenBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, QwenBlock)


class QwenBlock(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg

        self.ln_1 = LlamaRMSNorm(
            cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=device
        )
        self.attn = QwenAttention(cfg, device)
        self.ln_2 = LlamaRMSNorm(
            cfg.d_model, cfg.get("rms_norm_eps", 1e-6), device=device
        )
        self.mlp = QwenMLP(cfg, device)

    def prune_params(self, zs_block):
        self.attn.prune_params(zs_block)
        self.mlp.prune_params(zs_block)

        if self.attn.wq is None:
            self.ln_1 = None
        if self.mlp.gate_proj is None:
            self.ln_2 = None
        if "hidden_z" in zs_block:
            hidden_z = zs_block["hidden_z"]
            if self.ln_1 is not None:
                self.ln_1.prune_params(hidden_z)
            if self.ln_2 is not None:
                self.ln_2.prune_params(hidden_z)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.ByteTensor] = None,
        is_causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        retain_grad: bool = False,
        head_z: Optional[torch.Tensor] = None,
        head_layer_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        mlp_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None,
        qk_head_dim_z: Optional[torch.Tensor] = None,
        vo_head_dim_z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        if self.ln_1 is not None:
            a = self.ln_1(x, hidden_z=hidden_z)
            b, _, past_key_value = self.attn(
                a,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                attention_mask=attention_mask,
                retain_grad=retain_grad,
                head_z=head_z,
                head_layer_z=head_layer_z,
                hidden_z=hidden_z,
                qk_head_dim_z=qk_head_dim_z,
                vo_head_dim_z=vo_head_dim_z,
            )
        else:
            b = 0

        x = x + b

        if self.ln_2 is not None:
            m = self.ln_2(x, hidden_z=hidden_z)
            n = self.mlp(m, retain_grad, intermediate_z, mlp_z, hidden_z)
        else:
            n = 0

        x = x + n
        return x, past_key_value


class QwenAttention(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.attn_impl = cfg.attn_impl

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.d_model // self.n_heads
        self.total_head_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.pruned_heads = set()

        self.softmax_scale = cfg.get("softmax_scale")
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)

        self.wq = nn.Linear(self.d_model, self.d_model, device=device)
        self.wk = nn.Linear(
            self.d_model, self.n_kv_heads * self.head_dim, device=device
        )
        self.wv = nn.Linear(
            self.d_model, self.n_kv_heads * self.head_dim, device=device
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device, bias=False)

        self.attn_fn = flash_attn_fn if self.attn_impl == "flash" else normal_attn_fn
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=cfg.max_seq_len,
            base=cfg.get("rotary_emb_base", 1000000),
            device=device,
        )

    def prune_params(self, zs_block):
        # TODO
        pass

    def forward(
        self,
        x,
        past_key_value=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=True,
        needs_weights=False,
        attention_mask=None,
        retain_grad=False,
        head_z=None,
        head_layer_z=None,
        hidden_z=None,
        qk_head_dim_z=None,
        vo_head_dim_z=None,
    ):
        if self.wq is None:
            return None, None, past_key_value

        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)

        query_padding_mask = None
        if key_padding_mask is not None:
            query_padding_mask = key_padding_mask[:, -query.size(1) :]

        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1) :, -key.size(1) :]

        # b, s, d = query.shape
        query = rearrange(query, "b s (h d) -> b h s d", h=self.n_heads)
        key = rearrange(key, "b s (h d) -> b h s d", h=self.n_kv_heads)
        value = rearrange(value, "b s (h d) -> b h s d", h=self.n_kv_heads)

        key = key.repeat_interleave(self.n_kv_groups, dim=1)
        value = value.repeat_interleave(self.n_kv_groups, dim=1)

        kv_seq_len = key.size(2)
        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[-2]
            kv_seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, offset=offset)

        offset = 0
        if past_key_value is not None:
            if len(past_key_value) != 0:
                offset = past_key_value[0].shape[-2]
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
                past_key_value = (key, value)

        if self.attn_fn == flash_attn_fn:
            query = rearrange(query, "b h s d -> b s h d")
            key = rearrange(key, "b h s d -> b s h d")
            value = rearrange(value, "b h s d -> b s h d")
            context, attn_weights = self.attn_fn(
                query,
                key,
                value,
                softmax_scale=self.softmax_scale,
                attn_bias=attn_bias,
                query_padding_mask=query_padding_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                training=self.training,
                needs_weights=needs_weights,
                head_z=head_z,
            )
        else:
            context = self.attn_fn(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                head_z=head_z,
            )
            attn_weights = None

        if retain_grad:
            self.context = context
            if self.context.requires_grad:
                self.context.retain_grad()

        output = self.out_proj(context)

        # Apply layer-level mask
        if head_layer_z is not None:
            output *= head_layer_z

        # Apply hidden dimension mask
        if hidden_z is not None:
            output *= hidden_z

        if retain_grad:
            self.output = output
            if self.output.requires_grad:
                self.output.retain_grad()

        return output, attn_weights, past_key_value


class QwenMLP(nn.Module):
    def __init__(self, cfg: DictConfig, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.gate_proj = nn.Linear(
            cfg.d_model, cfg.intermediate_size, device=device, bias=False
        )
        self.up_proj = nn.Linear(
            cfg.d_model, cfg.intermediate_size, device=device, bias=False
        )
        self.down_proj = nn.Linear(
            cfg.intermediate_size, cfg.d_model, device=device, bias=False
        )

    def prune_params(self, zs_block):
        intermediate_z = zs_block.get("intermediate_z", None)
        mlp_z = zs_block.get("mlp_z", None)
        hidden_z = zs_block.get("hidden_z", None)

        # update params #
        if intermediate_z is not None:
            self.up_proj.weight.data = (
                self.up_proj.weight.data.transpose(0, 1)
                .mul(intermediate_z.squeeze(0))
                .transpose(0, 1)
            )
        if mlp_z is not None:
            self.down_proj.weight.data = (
                self.down_proj.weight.data.transpose(0, 1).mul(mlp_z).transpose(0, 1)
            )
        if hidden_z is not None:
            self.down_proj.weight.data = (
                self.down_proj.weight.data.transpose(0, 1).mul(hidden_z).transpose(0, 1)
            )
        #################

        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            print(f"    FFN hidden dim: {len(hidden_z)} -> {len(remaining_index)}")
            half = next(self.up_proj.parameters()).dtype
            self.up_proj = prune_linear_layer(self.up_proj, remaining_index, dim=1)
            self.gate_proj = prune_linear_layer(self.gate_proj, remaining_index, dim=1)
            self.down_proj = prune_linear_layer(self.down_proj, remaining_index, dim=0)
            if half == torch.float16:
                self.up_proj = self.up_proj.half()
                self.gate_proj = self.gate_proj.half()
                self.down_proj = self.down_proj.half()

        keep_dim = turn_mlp_z(intermediate_z, mlp_z)
        device = self.up_proj.weight.device
        if len(keep_dim) == self.up_proj.weight.shape[0]:
            print(
                f"    FFN intermediate dim: {self.cfg.intermediate_size} -> {len(keep_dim)}"
            )
            return

        if len(keep_dim) == 0:
            self.up_proj = None
            self.down_proj = None
            self.gate_proj = None
        else:
            keep_dim_index = torch.tensor(keep_dim).long().to(device)
            half = next(self.up_proj.parameters()).dtype
            self.up_proj = prune_linear_layer(self.up_proj, keep_dim_index, dim=0)
            self.gate_proj = prune_linear_layer(self.gate_proj, keep_dim_index, dim=0)
            self.down_proj = prune_linear_layer(self.down_proj, keep_dim_index, dim=1)
            if half == torch.float16:
                self.up_proj = self.up_proj.half()
                self.gate_proj = self.gate_proj.half()
                self.down_proj = self.down_proj.half()
        print(
            f"    FFN intermediate dim: {self.cfg.intermediate_size} -> {len(keep_dim)}"
        )

    def forward(
        self, x, retain_grad=False, intermediate_z=None, mlp_z=None, hidden_z=None
    ):
        if self.up_proj is None:
            return None
        gate = F.silu(self.gate_proj(x))
        up_v = self.up_proj(x)
        if retain_grad:
            self.up_v = up_v
            if self.up_v.requires_grad:
                self.up_v.retain_grad()
        if intermediate_z is not None:
            up_v *= intermediate_z
        down_v = self.down_proj(gate * up_v)

        if retain_grad:
            self.output = down_v
            if self.output.requires_grad:
                self.output.retain_grad()

        if mlp_z is not None:
            down_v = down_v * mlp_z

        if hidden_z is not None:
            down_v = down_v * hidden_z

        return down_v
