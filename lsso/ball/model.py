from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import CoreMode, LSSOConfig
from .reference import (
    accretive_equilibrium_mix,
    accretive_generator,
    bounded_complement,
    compact_equilibrium_diagnostics,
    qr_soft_frame,
    tensor_core_linear,
    tensor_core_matmul,
)


_CONTRACT_VERSION = 13
_ETA_INIT = 0.9
_ETA_INIT_RAW = math.atanh(_ETA_INIT)
_SUPPORTED_ACTIVATION_DTYPES = frozenset(
    (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)


class LSSO(nn.Module):
    r"""The accretive-equilibrium low-rank token mixer.

    Each head builds a QR soft frame P, compresses content into Z = P^T C,
    forms an accretive generator from static or cross-moment coordinates, and
    evaluates its reflected-resolvent equilibrium directly.

    Dynamic, static, and zero compact-core ownership share this one forward
    path. Position features, if needed, belong to the surrounding encoder.
    """

    def __init__(self, config: LSSOConfig) -> None:
        super().__init__()
        self.config = config
        self.w_bc = nn.Linear(
            config.dim,
            config.num_heads * config.rank + config.dim,
            bias=config.bias,
        )
        self.w_o = nn.Linear(config.dim, config.dim, bias=config.bias)

        if config.core_mode is CoreMode.DYNAMIC:
            self.core_base_raw = nn.Parameter(
                torch.zeros(config.num_heads, config.rank, config.rank)
            )
            self.core_drive_weight = nn.Parameter(
                torch.zeros(config.num_heads, config.head_dim, config.rank)
            )
        elif config.core_mode is CoreMode.STATIC:
            self.core_base_raw = nn.Parameter(
                torch.zeros(config.num_heads, config.rank, config.rank)
            )
            self.register_parameter("core_drive_weight", None)
        else:
            self.register_parameter("core_base_raw", None)
            self.register_parameter("core_drive_weight", None)

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
        }

    def get_extra_state(self) -> dict[str, object]:
        """Persist the exact operator contract with tensor state."""

        return self._contract_state()

    def set_extra_state(self, state: object) -> None:
        """Reject checkpoints created for a different operator contract."""

        expected = self._contract_state()
        if not isinstance(state, dict):
            raise RuntimeError("LSSO checkpoint is missing its configuration contract")
        if state != expected:
            keys = sorted(set(state) | set(expected))
            mismatches = ", ".join(
                f"{key}: checkpoint={state.get(key)!r}, model={expected.get(key)!r}"
                for key in keys
                if state.get(key) != expected.get(key)
            )
            raise RuntimeError(f"incompatible LSSO checkpoint contract ({mismatches})")

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
        if contract_key not in state_dict:
            error_msgs.append(
                "LSSO checkpoint is missing its configuration contract "
                f"({contract_key!r})"
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

    def _compact_coordinates(
        self,
        compact_state: torch.Tensor,
        valid_count: torch.Tensor,
    ) -> torch.Tensor | None:
        config = self.config
        if config.core_mode is CoreMode.DYNAMIC:
            if self.core_base_raw is None or self.core_drive_weight is None:
                raise RuntimeError("dynamic compact parameters are missing")
            dynamic_coordinates = tensor_core_matmul(
                compact_state,
                self.core_drive_weight.to(dtype=compact_state.dtype),
            )
            base = self.core_base_raw.to(dtype=compact_state.dtype).unsqueeze(0)
            return base + dynamic_coordinates / valid_count.sqrt().view(
                -1, 1, 1, 1
            )

        if config.core_mode is CoreMode.STATIC:
            if self.core_base_raw is None:
                raise RuntimeError("static compact parameters are missing")
            return self.core_base_raw.to(dtype=compact_state.dtype)

        return None

    def _forward_cuda(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the explicit strict native implementation of the default operator."""

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
        for name, parameter in (
            ("w_bc.weight", self.w_bc.weight),
            ("w_bc.bias", self.w_bc.bias),
            ("w_o.weight", self.w_o.weight),
            ("w_o.bias", self.w_o.bias),
            ("core_base_raw", self.core_base_raw),
            ("core_drive_weight", self.core_drive_weight),
            ("eta_raw", self.eta_raw),
        ):
            if parameter is None:
                continue
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
        if all_valid:
            valid_counts = None
        else:
            assert mask is not None
            valid_counts = (
                mask.sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0).contiguous()
            )

        # The native mixer consumes wide-range BF16 packed coordinates directly.
        if all_valid:
            safe_x = x
        else:
            assert mask is not None
            safe_x = torch.where(mask[:, :, None], x, torch.zeros_like(x))
        projected = tensor_core_linear(
            safe_x, self.w_bc.weight, self.w_bc.bias, output_dtype=torch.bfloat16
        )
        if not all_valid:
            assert mask is not None
            projected = torch.where(
                mask[:, :, None], projected, torch.zeros_like(projected)
            )
        if not projected.is_contiguous():
            raise RuntimeError(
                "implementation='cuda' requires w_bc to produce contiguous "
                "projected coordinates"
            )
        from . import cuda as cuda_backend

        mixed = cuda_backend.fast_mix(
            projected,
            self.core_base_raw if self.core_base_raw is not None else self.eta_raw.new_empty((config.num_heads, config.rank, 0)),
            self.core_drive_weight if self.core_drive_weight is not None else self.eta_raw.new_empty((config.num_heads, config.head_dim, 0)),
            self.eta_raw,
            valid_counts,
        )
        output = tensor_core_linear(
            mixed, self.w_o.weight, self.w_o.bias, output_dtype=x.dtype
        )
        if not all_valid:
            assert mask is not None
            output = torch.where(mask[:, :, None], output, torch.zeros_like(output))
        return output

    def _reference_compact_problem(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None,
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
            assert mask is not None
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
            compact_state = tensor_core_matmul(frame.mT, content)
            coordinates = self._compact_coordinates(compact_state, valid_count)
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
    ) -> dict[str, torch.Tensor]:
        """Return per-sample, per-head certificates for the realized operator."""

        self._validate_input(x)
        if self.config.core_mode is CoreMode.ZERO:
            raise ValueError("compact diagnostics require a nonzero compact core")
        _projected, frame, compact_state, _content, generator, eta, _mask = (
            self._reference_compact_problem(x, valid_mask)
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
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if x.dtype not in _SUPPORTED_ACTIVATION_DTYPES:
            raise TypeError(
                f"LSSO does not support x with dtype {x.dtype}; use "
                "torch.float16, torch.bfloat16, torch.float32, or torch.float64"
            )
        if x.shape[0] == 0:
            raise ValueError("batch size must be positive")
        if x.shape[1] == 0:
            raise ValueError("sequence length must be positive")

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        implementation: str = "reference",
    ) -> torch.Tensor:
        self._validate_input(x)
        config = self.config
        batch, length, _dim = x.shape

        if implementation == "cuda":
            return self._forward_cuda(x, valid_mask)
        if implementation != "reference":
            raise ValueError(
                "implementation must be 'reference' or 'cuda', "
                f"got {implementation!r}"
            )

        projected, frame, compact_state, content, generator, eta, mask = (
            self._reference_compact_problem(x, valid_mask)
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = accretive_equilibrium_mix(
                frame,
                generator,
                compact_state,
                content,
                eta,
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
        assert mask is not None
        return torch.where(mask[:, :, None], output, torch.zeros_like(output))

    def extra_repr(self) -> str:
        config = self.config
        return (
            f"dim={config.dim}, num_heads={config.num_heads}, rank={config.rank}, "
            f"core_mode={config.core_mode.value}, "
            "eta=per-head-interior-tanh"
        )


__all__ = ["LSSO"]
