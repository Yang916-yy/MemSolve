"""MemSolve-BERT: Hugging Face BERT with MemSolve in the attention sublayer.

BERT owns embeddings, Post-LN residuals and the tied MLM head; FFNs use SwiGLU.
MemSolve owns all mixer mathematics, relation/content projections, compact generator and output projection.
Import this module before using the HF Auto classes with MemSolve checkpoints.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertModel,
    BertOnlyMLMHead,
    BertPooler,
    BertPreTrainedModel,
    BertSelfOutput,
)

from memsolve import CoreMode, MemSolve, MemSolveConfig


class MemSolveBertConfig(BertConfig):
    """BERT geometry plus an explicit MemSolve backend and checkpoint contract."""

    model_type = "memsolve_bert"

    def __init__(
        self,
        memsolve_rank: int = 32,
        memsolve_core_mode: str = "equilibrium",
        memsolve_implementation: str = "reference",
        memsolve_contract: dict | None = None,
        memsolve_short_conv_kernel: int = 3,
        memsolve_rotary_base: float = 10000.0,
        memsolve_rotary_train_length: int = 128,
        memsolve_rotary_scaling: str = "yarn",
        ffn_type: str = "swiglu",
        **kwargs,
    ):
        if ffn_type != "swiglu":
            raise ValueError("MemSolve-BERT uses a SwiGLU FFN")
        # Each SwiGLU branch has width m; match a 4d FFN via 3dm = 8d².
        dim = kwargs.get("hidden_size", 768)
        kwargs.setdefault("num_hidden_layers", 14)
        kwargs.setdefault("intermediate_size", ((8 * dim + 47) // 48) * 16)
        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("position_embedding_type", "rove")
        # MemSolve has no materialized attention probabilities to drop out.
        kwargs.setdefault("attention_probs_dropout_prob", 0.0)
        super().__init__(**kwargs)
        MemSolveConfig(
            dim=self.hidden_size, num_heads=self.num_attention_heads,
            rank=memsolve_rank, bias=True,
            core_mode=memsolve_core_mode,
            rotary=self.position_embedding_type == "rove",
            rotary_base=memsolve_rotary_base,
            rotary_train_length=memsolve_rotary_train_length,
            rotary_scaling=memsolve_rotary_scaling,
        )
        if memsolve_implementation not in ("reference", "cuda"):
            raise ValueError("memsolve_implementation must be 'reference' or 'cuda'")
        if memsolve_implementation == "cuda" and memsolve_rank not in (16, 32, 48, 64):
            raise ValueError("CUDA MemSolve requires rank 16, 32, 48 or 64")
        if self.is_decoder or self.add_cross_attention:
            raise ValueError("MemSolve-BERT supports bidirectional encoders only")
        if self.position_embedding_type not in ("absolute", "rove", "none"):
            raise ValueError("position_embedding_type must be 'absolute', 'rove', or 'none'")
        if self.attention_probs_dropout_prob != 0:
            raise ValueError("Use hidden_dropout_prob; MemSolve has no attention-probability dropout")
        if memsolve_short_conv_kernel not in (0, 3):
            raise ValueError("memsolve_short_conv_kernel must be 0 (off) or 3 (centered)")
        self.memsolve_rank = memsolve_rank
        self.memsolve_core_mode = CoreMode(memsolve_core_mode).value
        self.memsolve_implementation = memsolve_implementation
        self.memsolve_contract = copy.deepcopy(memsolve_contract)
        self.memsolve_short_conv_kernel = memsolve_short_conv_kernel
        self.memsolve_rotary_base = memsolve_rotary_base
        self.memsolve_rotary_train_length = memsolve_rotary_train_length
        self.memsolve_rotary_scaling = memsolve_rotary_scaling
        self.ffn_type = ffn_type

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Saved configs include the operator contract written at construction.
        if not isinstance(config_dict.get("memsolve_contract"), dict):
            raise ValueError("MemSolve-BERT checkpoint config is missing memsolve_contract")
        if config_dict.get("ffn_type") != "swiglu":
            raise ValueError("MemSolve-BERT checkpoint must declare ffn_type='swiglu'")
        return super().from_dict(config_dict, **kwargs)


def _restore_core_contract(module, state_dict, prefix, *args):
    # Restore extra state from the validated HF config during parameter loading.
    state_dict.setdefault(prefix + "_extra_state", copy.deepcopy(module._hf_contract))


def _tensor_only_state_dict(module, state_dict, prefix, local_metadata):
    # HF stores the shared operator contract in config.json instead of tensors.
    for name, child in module.named_modules():
        if isinstance(child, MemSolve):
            state_dict.pop(prefix + name + "._extra_state", None)


class _CenteredShortConv(nn.Module):
    """Identity-initialized local residual using native depthwise Conv1d.

    Mask the input and output so padding stays zero across the convolution.
    """

    def __init__(self, dim, kernel):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)

    def forward(self, x, valid_mask, *, sequence_parallel=None):
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        if sequence_parallel is None:
            local = self.conv(x.transpose(1, 2)).transpose(1, 2)
        else:
            # The halo supplies neighboring tokens and global-end padding.
            padded = sequence_parallel.halo(x, self.conv.kernel_size[0] // 2)
            local = torch.nn.functional.conv1d(
                padded.transpose(1, 2), self.conv.weight, self.conv.bias,
                groups=self.conv.groups,
            ).transpose(1, 2)
        x = x + local
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask[..., None], 0)
        return x


class _MemSolveBertAttention(nn.Module):
    def __init__(self, config: MemSolveBertConfig):
        super().__init__()
        self.mixer = MemSolve(MemSolveConfig(
            dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            rank=config.memsolve_rank,
            bias=True,
            core_mode=config.memsolve_core_mode,
            rotary=config.position_embedding_type == "rove",
            rotary_base=config.memsolve_rotary_base,
            rotary_train_length=config.memsolve_rotary_train_length,
            rotary_scaling=config.memsolve_rotary_scaling,
        ))
        if config.memsolve_contract is None:
            config.memsolve_contract = self.mixer.get_extra_state()
        self.mixer.set_extra_state(config.memsolve_contract)
        self.mixer._hf_contract = copy.deepcopy(config.memsolve_contract)
        self.mixer.register_load_state_dict_pre_hook(_restore_core_contract)
        self.implementation = config.memsolve_implementation
        self.short_conv = (_CenteredShortConv(config.hidden_size, config.memsolve_short_conv_kernel)
                           if config.memsolve_short_conv_kernel else None)
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
        valid_mask = valid_counts = rotary_factors = None
        if isinstance(cache_position, tuple):
            valid_mask, valid_counts, rotary_factors = cache_position
            cache_position = None
        elif attention_mask is not None:
            # BERT's eager mask preparation produces additive [B, 1, 1, N].
            expected = (hidden_states.shape[0], 1, 1, hidden_states.shape[1])
            if attention_mask.shape != expected:
                raise ValueError("MemSolve-BERT accepts a 2D padding mask, not pairwise masks")
            valid_mask = attention_mask[:, 0, 0, :] == 0
        x = (self.short_conv(hidden_states, valid_mask)
             if self.short_conv is not None else hidden_states)
        if self.implementation == "cuda":
            # Load the device's native extension once for HF inference/training.
            from memsolve.ball import cuda
            cuda.load(device=x.device)
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        elif self.implementation == "cuda":
            # Plain HF inference uses BF16 activations with FP32 parameters.
            x = x.to(torch.bfloat16)
        mixed = self.mixer(x, valid_mask, implementation=self.implementation,
                           position_ids=cache_position,
                           valid_counts=valid_counts, rotary_factors=rotary_factors)
        return (self.output(mixed, hidden_states),)

    def prune_heads(self, heads):
        raise NotImplementedError("MemSolve-BERT does not support attention-head pruning")


class _MemSolveBertMixin:
    config_class = MemSolveBertConfig
    _supports_sdpa = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, _CenteredShortConv):
            nn.init.zeros_(module.conv.weight)
            nn.init.zeros_(module.conv.bias)
        if isinstance(module, MemSolve):
            with torch.no_grad():
                if module.core_base_raw is not None:
                    module.core_base_raw.zero_()
                import math
                module.eta_raw.fill_(math.atanh(0.9))

    @classmethod
    def _load_pretrained_model(cls, *args, **kwargs):
        result = super()._load_pretrained_model(*args, **kwargs)
        missing_keys = result[1]
        if any("attention.mixer." in key for key in missing_keys):
            raise RuntimeError("Checkpoint is missing MemSolve mixer weights; cannot load a stock BERT checkpoint")
        if any("intermediate.gate_up_proj." in key for key in missing_keys):
            raise RuntimeError("Checkpoint is missing SwiGLU gate/up weights")
        if any("attention.short_conv." in key for key in missing_keys):
            raise RuntimeError("Checkpoint is missing centered short-convolution weights")
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
        if config.position_embedding_type != "absolute":
            del self.embeddings.position_embeddings
            del self.embeddings.position_ids
            del self.embeddings.token_type_ids
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
        if self.position_embedding_type != "absolute" and position_ids is None:
            inputs = input_ids if input_ids is not None else inputs_embeds
            if inputs is not None:
                position_ids = torch.arange(inputs.shape[1], device=inputs.device).unsqueeze(0)
        # Like HF Llama, prepare position factors once and pass them through
        # every layer/checkpoint replay. No state survives this forward call.
        inputs = input_ids if input_ids is not None else inputs_embeds
        valid_mask = None if attention_mask is None else attention_mask.to(device=inputs.device, dtype=torch.bool)
        first = self.encoder.layer[0].attention
        valid_counts = (valid_mask.sum(-1, dtype=torch.float32).clamp_min_(1)
                        if valid_mask is not None and first.implementation == "cuda" else None)
        rotary_factors = None
        if self.position_embedding_type == "rove":
            shape = self.embeddings.word_embeddings.weight.new_empty((*inputs.shape[:2], 0))
            rotary_factors = first.mixer._rotary_factors(shape, valid_mask, position_ids, None)
        shared_metadata = (valid_mask, valid_counts, rotary_factors)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict,
            cache_position=shared_metadata,
        )


class MemSolveBertForMaskedLM(_MemSolveBertMixin, BertForMaskedLM):
    """MemSolve-BERT plus the unmodified Hugging Face tied MLM prediction head."""

    def __init__(self, config: MemSolveBertConfig):
        BertPreTrainedModel.__init__(self, config)
        self.bert = MemSolveBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.register_state_dict_post_hook(_tensor_only_state_dict)
        self.post_init()


class MemSolveBertForSequenceClassification(_MemSolveBertMixin, BertForSequenceClassification):
    """BERT's CLS pooler and classification/regression head for downstream tasks."""

    def __init__(self, config: MemSolveBertConfig):
        BertPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.bert = MemSolveBertModel(config, add_pooling_layer=True)
        dropout = config.classifier_dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob if dropout is None else dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.register_state_dict_post_hook(_tensor_only_state_dict)
        self.post_init()


AutoConfig.register(MemSolveBertConfig.model_type, MemSolveBertConfig)
AutoModel.register(MemSolveBertConfig, MemSolveBertModel)
AutoModelForMaskedLM.register(MemSolveBertConfig, MemSolveBertForMaskedLM)
AutoModelForSequenceClassification.register(MemSolveBertConfig, MemSolveBertForSequenceClassification)

__all__ = ["MemSolveBertConfig", "MemSolveBertModel", "MemSolveBertForMaskedLM",
           "MemSolveBertForSequenceClassification"]
