from __future__ import annotations

import torch
import torch.nn as nn

from .config import MemSolveConfig
from .reference import (
    ridge_query_readout, head_rms_norm, low_rank_gate_logits, sigmoid_output_gate,
    axial_rotary_tables, convolve_qk, rotary_qk, split_qkv, tensor_core_linear,
    readout_linear,
)


_CONTRACT_VERSION = 23
_SUPPORTED_ACTIVATION_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)


class MemSolve(nn.Module):
    """Independent key/value ridge memory with learned query readout.

    The learned core is shared across samples. The memory, key Gram and
    queries depend on the input. Centered depthwise convolutions give Q/K
    local context before the global solve, followed by axial 2D RoPE for
    patch grids. V remains a tokenwise projection.
    A low-rank sigmoid gate selects normalized readout channels before Wo.
    This mixer has no internal content skip.
    """

    def __init__(self, config: MemSolveConfig) -> None:
        super().__init__()
        self.config = config
        self.w_qkv = nn.Linear(
            config.dim,
            2 * config.num_heads * config.rank + config.dim,
            bias=config.bias,
        )
        qk_channels = 2 * config.num_heads * config.rank
        conv = nn.Conv1d if config.qk_conv_dim == 1 else nn.Conv2d
        self.qk_conv = conv(
            qk_channels, qk_channels, config.qk_conv_kernel_size,
            padding=config.qk_conv_kernel_size // 2, groups=qk_channels, bias=False,
        )
        self.core_delta = nn.Parameter(
            torch.zeros(config.num_heads, config.rank, config.rank)
        )
        # Gated DeltaNet/FLA output RMSNorm: one channel gain shared by all heads.
        self.head_norm_weight = nn.Parameter(torch.ones(config.head_dim))
        self.gate_down = nn.Linear(config.dim, config.output_gate_rank, bias=False)
        self.gate_up = nn.Linear(config.output_gate_rank, config.dim, bias=True)
        self.w_o = nn.Linear(config.dim, config.dim, bias=config.bias)
        # Fixed tables are derived state, excluded from checkpoints and DDP
        # buffer broadcasts. Rebuild on changes of grid, device or oracle dtype.
        # Keep each warmed geometry alive: a captured graph may still use its
        # tables after an eager evaluation at another resolution.
        self._rope_cache = {}
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize fan-in K and identity Q/K convolutions; leave Q/V alone.

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
        # Identity preserves the projection scale at initialization. The Q and
        # K filters are independent; all neighboring taps remain trainable.
        self.qk_conv.weight.zero_()
        center = self.config.qk_conv_kernel_size // 2
        self.qk_conv.weight[(slice(None), 0) + (center,) * self.config.qk_conv_dim] = 1
        # Keep ordinary Linear/framework weight initialization in both factors.
        # A zero bias centers the standard sigmoid near 0.5, not an identity gate.
        nn.init.zeros_(self.gate_up.bias)

    def _contract_state(self) -> dict[str, object]:
        config = self.config
        return {
            "version": _CONTRACT_VERSION,
            "operator": "qkv_ridge_query_readout",
            "dim": config.dim,
            "num_heads": config.num_heads,
            "rank": config.rank,
            "bias": config.bias,
            "qk_conv_dim": config.qk_conv_dim,
            "qk_conv_kernel_size": config.qk_conv_kernel_size,
            "qk_conv": "centered_depthwise_bias_free_identity_init",
            "output_gate_rank": config.output_gate_rank,
            "output_gate": "low_rank_linear_sigmoid_post_rms_pre_wo_with_bias",
            "position_encoding": (
                "rope_2d_axial_theta100_after_qk_conv" if config.qk_conv_dim == 2 else "external"
            ),
            "core": "identity_plus_learned_shared_query_delta",
            "readout": "query_memory_head_rmsnorm_shared_affine_eps_1e-6",
            "numerics": "bf16_qkv_fp16_readout_fp32_solve_v2",
        }

    def get_extra_state(self) -> dict[str, object]:
        return self._contract_state()

    def _rotary_tables(self, x, spatial_shape):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        key = (spatial_shape, x.device, dtype)
        if key not in self._rope_cache:
            self._rope_cache[key] = axial_rotary_tables(
                self.config.rank, spatial_shape, device=x.device, dtype=dtype,
            )
        return self._rope_cache[key]

    def set_extra_state(self, state: object) -> None:
        expected = self._contract_state()
        if not isinstance(state, dict):
            raise RuntimeError("MemSolve checkpoint is missing its configuration contract")
        if state != expected:
            keys = sorted(set(state) | set(expected))
            mismatches = ", ".join(
                f"{key}: checkpoint={state.get(key)!r}, model={expected.get(key)!r}"
                for key in keys
                if state.get(key) != expected.get(key)
            )
            raise RuntimeError(f"incompatible MemSolve checkpoint contract ({mismatches})")

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
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Framework adapters may restore serialized metadata in a PyTorch load
        # pre-hook. Check after those hooks; native checkpoints remain strict.
        if contract_key not in state_dict:
            error_msgs.append(
                f"MemSolve checkpoint is missing its configuration contract ({contract_key!r})"
            )

    @staticmethod
    def _validate_mask(valid_mask, *, batch, length, device):
        if valid_mask is None:
            return None
        if valid_mask.shape != (batch, length):
            raise ValueError(f"valid_mask must have shape {(batch, length)}")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        return valid_mask.to(device=device).contiguous()

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
        spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        self._validate_input(x)
        if implementation not in ("reference", "cuda"):
            raise ValueError("implementation must be 'reference' or 'cuda'")
        if implementation == "cuda":
            self._validate_cuda(x)
        config = self.config
        batch, length, _ = x.shape
        if config.qk_conv_dim == 2:
            if (not isinstance(spatial_shape, tuple) or len(spatial_shape) != 2
                    or any(not isinstance(s, int) or isinstance(s, bool) or s <= 0
                           for s in spatial_shape)
                    or spatial_shape[0] * spatial_shape[1] != length):
                raise ValueError("2D Q/K convolution requires spatial_shape=(H, W) with H*W=N")
        elif spatial_shape is not None:
            raise ValueError("spatial_shape is only supported for 2D Q/K convolution")
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
        cos, sin = self._rotary_tables(x, spatial_shape) if config.qk_conv_dim == 2 else (None, None)
        if implementation == "cuda":
            from . import cuda

            projected = cuda.local_qk(projected, self.qk_conv.weight, cos=cos, sin=sin,
                                      valid_mask=mask, spatial_shape=spatial_shape)
        else:
            projected = convolve_qk(projected, self.qk_conv.weight, spatial_shape)
            if config.qk_conv_dim == 2:
                projected = rotary_qk(projected, config.num_heads, cos, sin)
        if mask is not None and implementation == "reference":
            # Neighbors may write into an invalid position. It must not enter
            # either the key statistics or the readout, even with input bias.
            projected = torch.where(mask[..., None], projected, 0.0)
        # Only the compact map is materialized; dT/dDelta is the identity.
        core_map = self.core_delta + torch.eye(
            config.rank, device=self.core_delta.device, dtype=self.core_delta.dtype,
        )
        gate_logits = low_rank_gate_logits(
            safe_x, self.gate_down.weight,
            self.gate_up.weight, self.gate_up.bias,
        )
        if implementation == "cuda":
            from . import cuda

            output = cuda.fast_mix(
                projected, core_map, counts, norm_weight=self.head_norm_weight,
                gate_logits=gate_logits,
                output_weight=self.w_o.weight, output_bias=self.w_o.bias, output_dtype=x.dtype,
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
            normalized = self._gate_output(normalized, gate_logits)
            output = readout_linear(
                normalized, self.w_o.weight, self.w_o.bias, output_dtype=x.dtype
            )
        return output if mask is None else torch.where(mask[..., None], output, 0.0)

    def _gate_output(self, normalized, logits):
        # Pure PyTorch reference; the CUDA backend fuses this into its readout.
        return sigmoid_output_gate(normalized, logits)

    def extra_repr(self) -> str:
        c = self.config
        return (f"dim={c.dim}, num_heads={c.num_heads}, rank={c.rank}, "
                f"qk_conv_dim={c.qk_conv_dim}, qk_conv_kernel_size={c.qk_conv_kernel_size}, "
                f"output_gate_rank={c.output_gate_rank}, readout=qkv_ridge_query")


__all__ = ["MemSolve"]
