from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RidgonConfig:
    """Independent Q/K/V with a learned identity-plus-delta shared query map."""

    dim: int
    num_heads: int
    rank: int = 16
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("dim", "num_heads", "rank"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if not isinstance(self.bias, bool):
            raise TypeError("bias must be a bool")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


__all__ = ["RidgonConfig"]
