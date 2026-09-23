from __future__ import annotations

import torch
import torch.nn as nn

from .config import RidgonConfig
from .reference import (
    ridge_query_readout, head_rms_norm,
    split_qkv, tensor_core_linear,
)


_CONTRACT_VERSION = 19
_SUPPORTED_ACTIVATION_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)


class Ridgon(nn.Module):
    """Independent key/value ridge memory with learned query readout.

    The learned core is shared across samples. The memory, key Gram and
    queries depend on the input. Position features and residuals belong to
    the surrounding encoder; this mixer has no internal content skip.
    """

    def __init__(self, config: RidgonConfig) -> None:
        super().__init__()
        self.config = config
        self.w_qkv = nn.Linear(
            config.dim,
            2 * config.num_heads * config.rank + config.dim,
            bias=config.bias,
        )
        self.core_delta = nn.Parameter(
            torch.zeros(config.num_heads, config.rank, config.rank)
        )
        # Gated DeltaNet/FLA output RMSNorm: one channel gain shared by all heads.
        self.head_norm_weight = nn.Parameter(torch.ones(config.head_dim))
        self.w_o = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize K with fan-in variance; leave Q/V and other parameters alone.

        With unit-variance normalized inputs, Var(Wk)=1/dim gives expected
        trace(K^T K/n)/rank = 1, balancing key statistics with the unit ridge.
        timm's depth-first initializer calls this after initializing child
        Linear modules, so its generic std=0.02 does not override the K rule.
        """
        offset = self.config.num_heads * self.config.rank
        nn.init.normal_(
            self.w_qkv.weight[offset : 2 * offset],
            mean=0.0,
            std=self.config.dim ** -0.5,
        )
        if self.w_qkv.bias is not None:
            nn.init.zeros_(self.w_qkv.bias[offset : 2 * offset])

    def _contract_state(self) -> dict[str, object]:
        config = self.config
        return {
            "version": _CONTRACT_VERSION,
            "operator": "qkv_ridge_query_readout",
            "dim": config.dim,
            "num_heads": config.num_heads,
            "rank": config.rank,
            "bias": config.bias,
            "core": "identity_plus_learned_shared_query_delta",
            "readout": "query_memory_head_rmsnorm_shared_affine_eps_1e-6",
            "numerics": "bf16_projections_fp32_statistics_compact_v1",
        }

    def get_extra_state(self) -> dict[str, object]:
        return self._contract_state()

    def set_extra_state(self, state: object) -> None:
        expected = self._contract_state()
        if not isinstance(state, dict):
            raise RuntimeError("Ridgon checkpoint is missing its configuration contract")
        if state != expected:
            keys = sorted(set(state) | set(expected))
            mismatches = ", ".join(
                f"{key}: checkpoint={state.get(key)!r}, model={expected.get(key)!r}"
                for key in keys
                if state.get(key) != expected.get(key)
            )
            raise RuntimeError(f"incompatible Ridgon checkpoint contract ({mismatches})")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        contract_key = f"{prefix}_extra_state"
        if contract_key not in state_dict:
            error_msgs.append(
                f"Ridgon checkpoint is missing its configuration contract ({contract_key!r})"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _validate_mask(valid_mask, *, batch, length, device):
        if valid_mask is None:
            return None
        if valid_mask.shape != (batch, length):
            raise ValueError(f"valid_mask must have shape {(batch, length)}")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        return valid_mask.to(device=device)

    def _validate_input(self, x):
        if x.ndim != 3 or x.shape[-1] != self.config.dim:
            raise ValueError(f"x must have shape [B, N, {self.config.dim}]")
        if x.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
            raise TypeError("x must be a supported floating-point tensor")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("batch size and sequence length must be positive")

    def _validate_cuda(self, x):
        if x.device.type != "cuda":
            raise ValueError("implementation='cuda' requires x to be a CUDA tensor")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "implementation='cuda' requires float16 or bfloat16 activations"
            )
        if self.config.rank not in (16, 32, 48, 64):
            raise ValueError("implementation='cuda' requires rank in {16,32,48,64}")
        for name, parameter in self.named_parameters():
            if parameter.device != x.device or parameter.dtype != torch.float32:
                raise TypeError(
                    f"implementation='cuda' requires FP32 {name} on {x.device}"
                )
            if not parameter.is_contiguous():
                raise RuntimeError(f"implementation='cuda' requires contiguous {name}")

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        implementation: str = "reference",
    ) -> torch.Tensor:
        self._validate_input(x)
        if implementation not in ("reference", "cuda"):
            raise ValueError("implementation must be 'reference' or 'cuda'")
        if implementation == "cuda":
            self._validate_cuda(x)
        config = self.config
        batch, length, _ = x.shape
        mask = self._validate_mask(
            valid_mask, batch=batch, length=length, device=x.device
        )
        safe_x = x if mask is None else torch.where(mask[..., None], x, 0.0)
        projected = tensor_core_linear(
            safe_x,
            self.w_qkv.weight,
            self.w_qkv.bias,
            output_dtype=torch.float64 if x.dtype == torch.float64 else torch.bfloat16,
        )
        counts = None
        if mask is not None:
            projected = torch.where(mask[..., None], projected, 0.0)
            counts = (
                mask.sum(-1)
                .to(torch.float64 if x.dtype == torch.float64 else torch.float32)
                .clamp_min(1)
            )
        # Only the compact map is materialized; dT/dDelta is the identity.
        core_map = self.core_delta + torch.eye(
            config.rank, device=self.core_delta.device, dtype=self.core_delta.dtype,
        )
        if implementation == "cuda":
            from . import cuda

            normalized = cuda.fast_mix(
                projected, core_map, counts, norm_weight=self.head_norm_weight,
            )
        else:
            query, key, value = split_qkv(projected, config.num_heads, config.rank)
            mixed = ridge_query_readout(
                query, key, value, core_map, counts,
            ).transpose(1, 2)
            normalized = head_rms_norm(
                mixed, self.head_norm_weight,
            )
        normalized = normalized.reshape(batch, length, config.dim)
        if x.dtype != torch.float64:
            normalized = normalized.to(torch.bfloat16)
        output = tensor_core_linear(
            normalized, self.w_o.weight, self.w_o.bias, output_dtype=x.dtype
        )
        return output if mask is None else torch.where(mask[..., None], output, 0.0)

    def extra_repr(self) -> str:
        c = self.config
        return f"dim={c.dim}, num_heads={c.num_heads}, rank={c.rank}, readout=qkv_ridge_query"


__all__ = ["Ridgon"]
