"""MemSolve-BERT: Hugging Face BERT with MemSolve in the attention sublayer.

BERT owns embeddings, Post-LN residuals and the tied MLM head; FFNs use SwiGLU.
MemSolve owns all mixer mathematics, QKV, head RMSNorm and output projection.
Import this module before using the HF Auto classes with MemSolve checkpoints.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, BertConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertModel,
    BertOnlyMLMHead,
    BertPooler,
    BertPreTrainedModel,
    BertSelfOutput,
)

from memsolve import MemSolve, MemSolveConfig


class MemSolveBertConfig(BertConfig):
    """BERT geometry plus an explicit MemSolve backend and checkpoint contract."""

    model_type = "memsolve_bert"

    def __init__(
        self,
        memsolve_rank: int = 32,
        memsolve_qk_conv_kernel_size: int = 3,
        memsolve_output_gate_rank: int = 32,
        memsolve_implementation: str = "reference",
        memsolve_contract: dict | None = None,
        ffn_type: str = "swiglu",
        **kwargs,
    ):
        if ffn_type != "swiglu":
            raise ValueError("MemSolve-BERT uses a SwiGLU FFN")
        # intermediate_size is the width of EACH gated branch, not their sum.
        # Match the weights of a conventional 4d FFN: 3dm = 8d^2.
        dim = kwargs.get("hidden_size", 768)
        kwargs.setdefault("intermediate_size", ((8 * dim + 47) // 48) * 16)
        kwargs.setdefault("use_cache", False)
        # MemSolve has no materialized attention probabilities to drop out.
        kwargs.setdefault("attention_probs_dropout_prob", 0.0)
        super().__init__(**kwargs)
        MemSolveConfig(
            dim=self.hidden_size, num_heads=self.num_attention_heads,
            rank=memsolve_rank, bias=True,
            qk_conv_kernel_size=memsolve_qk_conv_kernel_size,
            output_gate_rank=memsolve_output_gate_rank,
        )
        if memsolve_implementation not in ("reference", "cuda"):
            raise ValueError("memsolve_implementation must be 'reference' or 'cuda'")
        if memsolve_implementation == "cuda" and memsolve_rank not in (16, 32, 48, 64):
            raise ValueError("CUDA MemSolve requires rank 16, 32, 48 or 64")
        if self.is_decoder or self.add_cross_attention:
            raise ValueError("MemSolve-BERT supports bidirectional encoders only")
        if self.position_embedding_type != "absolute":
            raise ValueError("MemSolve-BERT currently uses BERT absolute position embeddings")
        if self.attention_probs_dropout_prob != 0:
            raise ValueError("Use hidden_dropout_prob; MemSolve has no attention-probability dropout")
        self.memsolve_rank = memsolve_rank
        self.memsolve_qk_conv_kernel_size = memsolve_qk_conv_kernel_size
        self.memsolve_output_gate_rank = memsolve_output_gate_rank
        self.memsolve_implementation = memsolve_implementation
        self.memsolve_contract = copy.deepcopy(memsolve_contract)
        self.ffn_type = ffn_type

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # A newly constructed config acquires its contract from the actual core.
        # A saved config must already have one; never infer a checkpoint version.
        if not isinstance(config_dict.get("memsolve_contract"), dict):
            raise ValueError("MemSolve-BERT checkpoint config is missing memsolve_contract")
        if config_dict.get("ffn_type") != "swiglu":
            raise ValueError("MemSolve-BERT checkpoint must declare ffn_type='swiglu'")
        return super().from_dict(config_dict, **kwargs)


def _restore_core_contract(module, state_dict, prefix, *args):
    # HF 4.57 loads individual parameters with assign=True. Its config has
    # already been validated against the core in _MemSolveBertAttention.__init__.
    state_dict.setdefault(prefix + "_extra_state", copy.deepcopy(module._hf_contract))


def _tensor_only_state_dict(module, state_dict, prefix, local_metadata):
    # The HF checkpoint format stores the common operator contract in config.json.
    # Standalone MemSolve.state_dict() continues to use its native extra-state dict.
    for name, child in module.named_modules():
        if isinstance(child, MemSolve):
            state_dict.pop(prefix + name + "._extra_state", None)


class _MemSolveBertAttention(nn.Module):
    def __init__(self, config: MemSolveBertConfig):
        super().__init__()
        self.mixer = MemSolve(MemSolveConfig(
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            rank=config.memsolve_rank,
            bias=True,
            qk_conv_kernel_size=config.memsolve_qk_conv_kernel_size,
            output_gate_rank=config.memsolve_output_gate_rank,
        ))
        if config.memsolve_contract is None:
            config.memsolve_contract = self.mixer.get_extra_state()
        self.mixer.set_extra_state(config.memsolve_contract)
        self.mixer._hf_contract = copy.deepcopy(config.memsolve_contract)
        self.mixer.register_load_state_dict_pre_hook(_restore_core_contract)
        self.implementation = config.memsolve_implementation
        self.output = BertSelfOutput(config)
        # MemSolve already owns Wo. Keep only BERT's dropout + residual + LN.
        self.output.dense = nn.Identity()

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None,
        encoder_hidden_states=None, past_key_values=None,
        output_attentions=False, cache_position=None,
    ):
        if head_mask is not None or output_attentions:
            raise ValueError("MemSolve does not expose attention probabilities or head masks")
        if encoder_hidden_states is not None or past_key_values is not None:
            raise ValueError("MemSolve-BERT does not support cross attention or KV caches")
        valid_mask = None
        if attention_mask is not None:
            # BERT's eager mask preparation produces additive [B, 1, 1, N].
            expected = (hidden_states.shape[0], 1, 1, hidden_states.shape[1])
            if attention_mask.shape != expected:
                raise ValueError("MemSolve-BERT accepts a 2D padding mask, not pairwise masks")
            valid_mask = attention_mask[:, 0, 0, :] == 0
        x = hidden_states
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        elif self.implementation == "cuda":
            # Also support plain HF inference, which does not enable autocast.
            # Operator parameters stay FP32; the activation boundary is BF16.
            x = x.to(torch.bfloat16)
        mixed = self.mixer(x, valid_mask, implementation=self.implementation)
        return (self.output(mixed, hidden_states),)

    def prune_heads(self, heads):
        raise NotImplementedError("MemSolve-BERT does not support attention-head pruning")


class _MemSolveBertMixin:
    config_class = MemSolveBertConfig
    _supports_sdpa = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, MemSolve):
            with torch.no_grad():
                module.core_delta.zero_()
                module.head_norm_weight.fill_(1.0)
                # HF initializes child linears first. Restore the core's K rule.
                module.init_weights()

    @classmethod
    def _load_pretrained_model(cls, *args, **kwargs):
        result = super()._load_pretrained_model(*args, **kwargs)
        missing_keys = result[1]
        if any("attention.mixer." in key for key in missing_keys):
            raise RuntimeError("Checkpoint is missing MemSolve mixer weights; cannot load a stock BERT checkpoint")
        if any("intermediate.gate_up_proj." in key for key in missing_keys):
            raise RuntimeError("Checkpoint is missing SwiGLU gate/up weights")
        return result


class _BertSwiGLUIntermediate(nn.Module):
    """Packed gate/up projection; BERT's output module owns the down projection.

    Uses the packed gate/up layout of Transformers' Phi3MLP, retaining BERT
    projection biases and its existing FFN output dropout/residual/LayerNorm.
    """

    def __init__(self, config: MemSolveBertConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size)

    def forward(self, hidden_states):
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return nn.functional.silu(gate) * up


class MemSolveBertModel(_MemSolveBertMixin, BertModel):
    """BERT encoder with a complete MemSolve mixer in every attention sublayer.

    The default has no pooler, as required for MLM and token-mean embeddings.
    Pass add_pooling_layer=True only when a downstream task needs BERT's pooler.
    """

    def __init__(self, config: MemSolveBertConfig, add_pooling_layer: bool = False):
        BertPreTrainedModel.__init__(self, config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        for layer in self.encoder.layer:
            layer.attention = _MemSolveBertAttention(config)
            layer.intermediate = _BertSwiGLUIntermediate(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.attn_implementation = "eager"
        self.position_embedding_type = config.position_embedding_type
        self.register_state_dict_post_hook(_tensor_only_state_dict)
        self.post_init()

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        past_key_values=None, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None, cache_position=None,
    ):
        if encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise ValueError("MemSolve-BERT does not support cross attention")
        if past_key_values is not None or use_cache or cache_position is not None:
            raise ValueError("MemSolve-BERT does not support KV caches")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("MemSolve-BERT accepts a 2D padding mask, not pairwise masks")
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
        )


class MemSolveBertForMaskedLM(_MemSolveBertMixin, BertForMaskedLM):
    """MemSolve-BERT plus the unmodified Hugging Face tied MLM prediction head."""

    def __init__(self, config: MemSolveBertConfig):
        BertPreTrainedModel.__init__(self, config)
        self.bert = MemSolveBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.register_state_dict_post_hook(_tensor_only_state_dict)
        self.post_init()


AutoConfig.register(MemSolveBertConfig.model_type, MemSolveBertConfig)
AutoModel.register(MemSolveBertConfig, MemSolveBertModel)
AutoModelForMaskedLM.register(MemSolveBertConfig, MemSolveBertForMaskedLM)

__all__ = ["MemSolveBertConfig", "MemSolveBertModel", "MemSolveBertForMaskedLM"]
