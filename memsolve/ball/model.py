from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import CoreMode, MemSolveConfig
from .reference import (
    _compact_matmul,
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    compact_equilibrium_diagnostics,
    qr_soft_frame,
    rotary_position_factors,
    rotate_content,
    tensor_core_linear,
    tensor_core_matmul,
)


_CONTRACT_VERSION = 26
_ETA_INIT = 0.9
_ETA_INIT_RAW = math.atanh(_ETA_INIT)
_SUPPORTED_ACTIVATION_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)


class MemSolve(nn.Module):
    r"""The accretive-equilibrium low-rank token mixer.

    Each head builds a QR soft frame P, compresses content into Z = P^T C,
    forms a learned per-head accretive generator, and
    evaluates its reflected-resolvent equilibrium directly.

    Equilibrium and complement share this forward path. Optional rotary value
    transport supplies relative position without changing the compact core.
    """

    def __init__(self, config: MemSolveConfig) -> None:
        super().__init__()
        self.config = config
        self.w_bc = nn.Linear(
            config.dim,
            config.num_heads * config.rank + config.dim,
            bias=config.bias,
        )
        self.w_o = nn.Linear(config.dim, config.dim, bias=config.bias)

        if config.core_mode is CoreMode.EQUILIBRIUM:
            self.core_base_raw = nn.Parameter(
                torch.zeros(config.num_heads, config.rank, config.rank)
            )
        else:
            self.register_parameter("core_base_raw", None)

        self.eta_raw = nn.Parameter(
            torch.full(
                (config.num_heads,),
                _ETA_INIT_RAW if config.scalar_complement else 0.0,
                dtype=torch.float32,
            ),
            requires_grad=config.scalar_complement,
        )

    def complement(self) -> torch.Tensor:
        """Return the learned per-head complement."""

        if not self.config.scalar_complement:
            return torch.zeros_like(self.eta_raw)
        return bounded_complement(self.eta_raw)

    def _contract_state(self) -> dict[str, object]:
        config = self.config
        return {
            "version": _CONTRACT_VERSION,
            "operator": "accretive_equilibrium",
            "dim": config.dim,
            "num_heads": config.num_heads,
            "rank": config.rank,
            "core_mode": config.core_mode.value,
            "eta_parameterization": "per_head_interior_tanh",
            "numerics": "tf32-wbc-ieee-fgram-tc16-v6",
            "bias": config.bias,
            **({"skew_coupling": False} if not config.skew_coupling else {}),
            **({"scalar_complement": False} if not config.scalar_complement else {}),
            **({"rotary": {
                "type": "value_transport",
                "layout": "full_head_adjacent_pairs",
                "base": config.rotary_base,
                "train_length": config.rotary_train_length,
                "scaling": config.rotary_scaling,
                "yarn_beta": [32, 1],
                "amplitude": 1.0,
            }} if config.rotary else {}),
        }

    def get_extra_state(self) -> dict[str, object]:
        """Persist the exact operator contract with tensor state."""

        return self._contract_state()

    def set_extra_state(self, state: object) -> None:
        """Reject checkpoints created for a different operator contract."""

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
        state_dict: dict[str, object],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Require the semantic contract even for a non-strict tensor load."""

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

        if contract_key not in state_dict:
            error_msgs.append(
                "MemSolve checkpoint is missing its configuration contract "
                f"({contract_key!r})"
            )

    @staticmethod
    def _validate_mask(
        valid_mask: torch.Tensor | None,
        *,
        batch: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(batch, length, dtype=torch.bool, device=device)
        if valid_mask.shape != (batch, length):
            raise ValueError(
                f"valid_mask must have shape {(batch, length)}, "
                f"got {tuple(valid_mask.shape)}"
            )
        if valid_mask.dtype != torch.bool:
            raise TypeError(
                f"valid_mask must have dtype torch.bool, got {valid_mask.dtype}"
            )
        return valid_mask.to(device=device)

    def _forward_cuda(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        sequence_parallel=None,
        valid_counts: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        rotary_context_length: int | torch.Tensor | None = None,
        rotary_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Project tokens, apply the CUDA mixer, and project back to model width."""

        config = self.config
        if (
            not config.skew_coupling
            or not config.scalar_complement
            or config.rank not in (16, 32, 48, 64)
        ):
            raise ValueError(
                "implementation='cuda' requires skew_coupling=True, scalar_complement=True, "
                "and rank in {16, 32, 48, 64}"
            )
        if x.device.type != "cuda":
            raise ValueError("implementation='cuda' requires x to be a CUDA tensor")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "implementation='cuda' supports x with dtype torch.float16 or "
                "torch.bfloat16; "
                f"got {x.dtype}"
            )

        for name, parameter in self.named_parameters():
            if parameter.device != x.device:
                raise RuntimeError(
                    f"implementation='cuda' requires {name} on {x.device}, "
                    f"got {parameter.device}"
                )
            if parameter.dtype != torch.float32:
                raise TypeError(
                    f"implementation='cuda' requires {name} to use torch.float32, "
                    f"got {parameter.dtype}"
                )
            if not parameter.is_contiguous():
                raise RuntimeError(
                    f"implementation='cuda' requires contiguous {name}"
                )

        batch, length, _dim = x.shape
        all_valid = valid_mask is None
        mask = (
            None
            if all_valid
            else self._validate_mask(
                valid_mask,
                batch=batch,
                length=length,
                device=x.device,
            )
        )
        if sequence_parallel is not None:
            sequence_parallel.validate(length, config.rank)
        rotary_cos, rotary_sin = rotary_factors if rotary_factors is not None else self._rotary_factors(
            x, mask, position_ids, rotary_context_length,
            sequence_parallel=sequence_parallel,
        )
        if valid_counts is None:
            if sequence_parallel is not None:
                valid_counts = sequence_parallel.valid_counts(mask, batch_size=batch, device=x.device)
            elif not all_valid:
                valid_counts = mask.sum(dim=-1, dtype=torch.float32).clamp_min_(1)
        elif (valid_counts.shape != (batch,) or valid_counts.device != x.device
              or valid_counts.dtype != torch.float32 or valid_counts.requires_grad
              or not valid_counts.is_contiguous()):
            raise ValueError("valid_counts must be non-differentiable contiguous FP32 [batch] on the input device")

        # The native mixer consumes wide-range BF16 packed coordinates directly.
        projected = tensor_core_linear(
            x, self.w_bc.weight, self.w_bc.bias, output_dtype=torch.bfloat16,
            fuse_weight_bias_grad=True, output_mask=mask, input_mask=mask,
        )
        from . import cuda as cuda_backend

        mixed = cuda_backend.fast_mix(
            projected,
            self.core_base_raw if self.core_base_raw is not None else self.eta_raw.new_empty((config.num_heads, config.rank, 0)),
            self.eta_raw,
            valid_counts,
            sequence_parallel=sequence_parallel,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
        )
        output = tensor_core_linear(
            mixed, self.w_o.weight, self.w_o.bias, output_dtype=x.dtype,
            fuse_weight_bias_grad=True, output_mask=mask,
        )
        return output

    def _reference_compact_problem(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        rotary_cos: torch.Tensor | None = None,
        rotary_sin: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Construct the canonical compact problem for reference evaluation."""

        config = self.config
        batch, length, _dim = x.shape
        all_valid = valid_mask is None
        mask = (
            None
            if all_valid
            else self._validate_mask(
                valid_mask,
                batch=batch,
                length=length,
                device=x.device,
            )
        )
        safe_x = (
            x
            if all_valid
            else torch.where(mask[:, :, None], x, torch.zeros_like(x))
        )
        projected = tensor_core_linear(
            safe_x,
            self.w_bc.weight,
            self.w_bc.bias,
            output_dtype=torch.float64 if x.dtype == torch.float64 else torch.bfloat16,
        )
        relation, content = projected.split(
            (config.num_heads * config.rank, config.dim), dim=-1
        )
        relation = relation.view(
            batch, length, config.num_heads, config.rank
        ).transpose(1, 2)
        content = content.view(
            batch, length, config.num_heads, config.head_dim
        ).transpose(1, 2)
        if not all_valid:
            active = mask[:, None, :, None]
            relation = torch.where(active, relation, torch.zeros_like(relation))
            content = torch.where(active, content, torch.zeros_like(content))

        calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            relation = relation.to(dtype=calc_dtype)
            content = content.to(dtype=calc_dtype)
            valid_count = (
                torch.full(
                    (batch,),
                    float(length),
                    dtype=calc_dtype,
                    device=x.device,
                )
                if all_valid
                else mask.sum(dim=-1).to(dtype=calc_dtype).clamp_min(1.0)
            )
            relation = relation / valid_count.sqrt().view(batch, 1, 1, 1)
            frame = qr_soft_frame(relation)
            transported = (
                content if rotary_cos is None
                else rotate_content(content, rotary_cos, rotary_sin)
            )
            compact_state = (
                tensor_core_matmul(frame.mT, transported) if rotary_cos is None
                else _compact_matmul(frame.mT, transported)
            )
            coordinates = (
                None if self.core_base_raw is None
                else self.core_base_raw.to(dtype=calc_dtype)
            )
            generator = (
                None if coordinates is None else accretive_generator(
                    coordinates, skew_coupling=config.skew_coupling
                )
            )
            eta = self.complement().to(device=x.device, dtype=calc_dtype)
        return projected, frame, compact_state, content, generator, eta, mask

    @torch.no_grad()
    def diagnostics(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        adjoint_rhs: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        rotary_context_length: int | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-sample, per-head certificates for the realized operator."""

        self._validate_input(x)
        if self.config.core_mode is CoreMode.COMPLEMENT:
            raise ValueError("compact diagnostics require equilibrium mode")
        rotary_cos, rotary_sin = self._rotary_factors(
            x, valid_mask, position_ids, rotary_context_length
        )
        _projected, frame, compact_state, _content, generator, eta, _mask = (
            self._reference_compact_problem(
                x, valid_mask, rotary_cos=rotary_cos, rotary_sin=rotary_sin
            )
        )
        assert generator is not None
        return compact_equilibrium_diagnostics(
            frame,
            generator,
            compact_state,
            eta,
            adjoint_rhs.to(device=x.device, dtype=compact_state.dtype),
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        config = self.config
        if x.ndim != 3 or x.shape[-1] != config.dim:
            raise ValueError(
                f"x must have shape [B, N, {config.dim}], got {tuple(x.shape)}"
            )
        if x.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
            raise TypeError(
                f"MemSolve does not support x with dtype {x.dtype}; use a floating-point tensor in "
                "torch.float16, torch.bfloat16, torch.float32, or torch.float64"
            )
        if x.shape[0] == 0:
            raise ValueError("batch size must be positive")
        if x.shape[1] == 0:
            raise ValueError("sequence length must be positive")

    def _rotary_factors(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        context_length: int | torch.Tensor | None,
        *,
        sequence_parallel=None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        config = self.config
        if not config.rotary:
            if position_ids is not None or context_length is not None:
                raise ValueError("position_ids and rotary_context_length require rotary=True")
            return None, None
        batch, length, _ = x.shape
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if position_ids is None:
            offset = 0 if sequence_parallel is None else sequence_parallel.offset
            position_ids = torch.arange(offset, offset + length, device=x.device)[None]
        else:
            if not isinstance(position_ids, torch.Tensor) or position_ids.dtype not in integer_dtypes:
                raise TypeError("position_ids must be an integer tensor")
            if position_ids.ndim == 1:
                position_ids = position_ids[None]
            if (position_ids.ndim != 2 or position_ids.shape[1] != length
                    or position_ids.shape[0] not in (1, batch)):
                raise ValueError("position_ids must have shape [N], [1,N], or [B,N]")
            position_ids = position_ids.to(device=x.device)
        if context_length is None:
            physical_length = length if sequence_parallel is None else sequence_parallel.global_length
            if (valid_mask is None or config.rotary_scaling == "none"
                    or physical_length <= config.rotary_train_length):
                # Every possible valid span has scale one within the training
                # context; keep a shared table instead of expanding over B.
                context_length = physical_length
            elif sequence_parallel is not None:
                raise ValueError("masked sequence parallel rotary requires explicit global rotary_context_length")
            else:
                mask = self._validate_mask(valid_mask, batch=batch, length=length, device=x.device)
                index = torch.arange(length, device=x.device)[None]
                first = torch.where(mask, index, length).amin(dim=-1)
                last = torch.where(mask, index, -1).amax(dim=-1)
                context_length = (last - first + 1).clamp_min(1)
        if isinstance(context_length, torch.Tensor):
            if context_length.dtype not in integer_dtypes or context_length.requires_grad:
                raise TypeError("rotary_context_length must be a non-differentiable integer tensor")
            if context_length.shape != (batch,):
                raise ValueError("rotary_context_length tensor must have shape [B]")
            context_length = context_length.to(device=x.device)
            torch._assert_async((context_length > 0).all(), "rotary_context_length must be positive")
        elif not isinstance(context_length, int) or isinstance(context_length, bool):
            raise TypeError("rotary_context_length must be an integer or integer tensor [B]")
        elif context_length <= 0:
            raise ValueError("rotary_context_length must be positive")
        return rotary_position_factors(
            position_ids, head_dim=config.head_dim, base=config.rotary_base,
            train_length=config.rotary_train_length, context_length=context_length,
            scaling=config.rotary_scaling, dtype=x.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        implementation: str = "reference",
        sequence_parallel=None,
        valid_counts: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        rotary_context_length: int | torch.Tensor | None = None,
        rotary_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._validate_input(x)
        config = self.config
        if rotary_factors is not None and (not config.rotary or position_ids is not None or rotary_context_length is not None):
            raise ValueError("precomputed rotary_factors require rotary=True and replace position/context inputs")
        batch, length, _dim = x.shape

        if implementation == "cuda":
            return self._forward_cuda(
                x, valid_mask, sequence_parallel=sequence_parallel, valid_counts=valid_counts,
                position_ids=position_ids, rotary_context_length=rotary_context_length,
                rotary_factors=rotary_factors,
            )
        if implementation != "reference":
            raise ValueError(
                "implementation must be 'reference' or 'cuda', "
                f"got {implementation!r}"
            )

        if sequence_parallel is not None or valid_counts is not None:
            raise ValueError("sequence_parallel and precomputed valid_counts require implementation='cuda'")

        rotary_cos, rotary_sin = rotary_factors if rotary_factors is not None else self._rotary_factors(
            x, valid_mask, position_ids, rotary_context_length
        )
        projected, frame, compact_state, content, generator, eta, mask = (
            self._reference_compact_problem(
                x, valid_mask, rotary_cos=rotary_cos, rotary_sin=rotary_sin
            )
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = accretive_equilibrium_mix(
                frame,
                generator,
                compact_state,
                content,
                eta,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
            )

        if x.dtype != torch.float64:
            output = output.to(torch.bfloat16)
        output = output.transpose(1, 2).contiguous().view(
            batch, length, config.dim
        )
        output = tensor_core_linear(
            output, self.w_o.weight, self.w_o.bias, output_dtype=x.dtype
        )
        if mask is None:
            return output
        return torch.where(mask[:, :, None], output, torch.zeros_like(output))

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"dim={config.dim}, num_heads={config.num_heads}, rank={config.rank}, "
            f"core_mode={config.core_mode.value}, "
            f"eta=per-head-interior-tanh, rotary={config.rotary}"
        )


__all__ = ["MemSolve"]
