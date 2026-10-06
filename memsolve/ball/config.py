from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemSolveConfig:
    """Q/K/V ridge readout with centered Q/K filters and a sigmoid output gate."""

    dim: int
    num_heads: int
    rank: int = 16
    bias: bool = False
    qk_conv_dim: int = 1
    qk_conv_kernel_size: int = 3
    output_gate_rank: int = 32

    def __post_init__(self) -> None:
        for name in ("dim", "num_heads", "rank", "qk_conv_kernel_size", "output_gate_rank"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if not isinstance(self.bias, bool):
            raise TypeError("bias must be a bool")
        if not isinstance(self.qk_conv_dim, int) or isinstance(self.qk_conv_dim, bool):
            raise TypeError("qk_conv_dim must be an integer")
        if self.qk_conv_dim not in (1, 2):
            raise ValueError("qk_conv_dim must be 1 (sequence) or 2 (patch grid)")
        if self.qk_conv_kernel_size % 2 != 1:
            raise ValueError("qk_conv_kernel_size must be odd for centered convolution")
        if self.qk_conv_dim == 2 and self.rank % 4:
            raise ValueError("2D axial RoPE requires rank divisible by 4")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


__all__ = ["MemSolveConfig"]
