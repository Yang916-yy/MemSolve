from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class CoreMode(str, Enum):
    """Compact-core modes supported by the one MemSolve operator."""

    EQUILIBRIUM = "equilibrium"
    COMPLEMENT = "complement"


@dataclass(frozen=True, slots=True)
class MemSolveConfig:
    """Configuration for the one MemSolve operator and its explicit ablations."""

    dim: int
    num_heads: int
    rank: int = 16
    core_mode: CoreMode = CoreMode.EQUILIBRIUM
    bias: bool = False
    skew_coupling: bool = True
    scalar_complement: bool = True
    rotary: bool = False
    rotary_base: float = 10000.0
    rotary_train_length: int = 128
    rotary_scaling: str = "yarn"

    def __post_init__(self) -> None:
        for name, value in (
            ("dim", self.dim),
            ("num_heads", self.num_heads),
            ("rank", self.rank),
            ("rotary_train_length", self.rotary_train_length),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )

        try:
            core_mode = CoreMode(self.core_mode)
        except (TypeError, ValueError) as error:
            choices = ", ".join(mode.value for mode in CoreMode)
            raise ValueError(
                f"core_mode must be one of {{{choices}}}, got {self.core_mode!r}"
            ) from error
        object.__setattr__(self, "core_mode", core_mode)

        if not isinstance(self.bias, bool):
            raise TypeError(f"bias must be a bool, got {type(self.bias).__name__}")

        for name in ("skew_coupling", "scalar_complement", "rotary"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

        if self.dim <= 0 or self.num_heads <= 0 or self.rank <= 0:
            raise ValueError("dim, num_heads, and rank must be positive")
        if self.dim % self.num_heads:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}"
            )
        if self.rotary_train_length <= 0:
            raise ValueError("rotary_train_length must be positive")
        if (not isinstance(self.rotary_base, (int, float))
                or isinstance(self.rotary_base, bool)):
            raise TypeError("rotary_base must be a real number")
        if not math.isfinite(self.rotary_base) or self.rotary_base <= 1:
            raise ValueError("rotary_base must be finite and greater than one")
        if self.rotary_scaling not in ("none", "yarn"):
            raise ValueError("rotary_scaling must be 'none' or 'yarn'")
        if self.rotary and self.head_dim % 2:
            raise ValueError("rotary requires an even head_dim")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


__all__ = ["CoreMode", "MemSolveConfig"]
