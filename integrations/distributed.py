"""Exact sequence-sharding utilities; model equations remain in memsolve.ball.

Each rank owns consecutive tokens of the same samples and replicated weights.
Use SUM (not DDP's averaging over this group) for replicated parameter gradients.
Create the context outside CUDA Graph capture, once for each shard layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class SequenceParallel:
    group: object
    shard_lengths: tuple[int, ...]
    rank: int

    @classmethod
    def create(cls, local_length: int, group=None):
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before sequence parallelism")
        lengths = [None] * dist.get_world_size(group)
        dist.all_gather_object(lengths, int(local_length), group=group)
        if any(length < 1 for length in lengths):
            raise ValueError("sequence shards must be nonempty")
        return cls(group, tuple(lengths), dist.get_rank(group))

    @property
    def world_size(self):
        return len(self.shard_lengths)

    @property
    def global_length(self):
        return sum(self.shard_lengths)

    @property
    def offset(self):
        return sum(self.shard_lengths[:self.rank])

    def validate(self, local_length, operator_rank):
        if local_length != self.shard_lengths[self.rank]:
            raise ValueError("local sequence length differs from the sequence-parallel layout")
        if self.global_length <= operator_rank:
            raise ValueError("sequence parallelism requires global sequence length > rank; use an unsharded short sequence")

    def sum_(self, tensor):
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return tensor

    def sum_tensors(self, *tensors):
        if self.world_size == 1:
            return tensors
        # One compact collective, with views rather than unpacking copies.
        flat = torch.cat([value.reshape(-1) for value in tensors])
        self.sum_(flat)
        pieces = flat.split([value.numel() for value in tensors])
        return tuple(piece.view_as(value) for piece, value in zip(pieces, tensors))

    def valid_counts(self, mask, *, batch_size=None, device=None):
        """Compute once per batch, reuse at every layer; clamp only after SUM."""
        if mask is None:
            if batch_size is None or device is None:
                raise ValueError("unmasked counts require batch_size and device")
            counts = torch.full((batch_size,), float(self.shard_lengths[self.rank]), device=device, dtype=torch.float32)
        else:
            if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[1] != self.shard_lengths[self.rank]:
                raise ValueError("mask must be boolean [batch, local_length]")
            counts = mask.sum(-1, dtype=torch.float32)
        # A fully valid local shard may omit its mask while another rank pads.
        # Every rank must still participate in this one metadata collective.
        return self.sum_(counts).clamp_min_(1)

    @torch.no_grad()
    def reduce_gradients_(self, module):
        """SUM once after accumulated backward, before clipping/optimizer.step.

        Do not also apply DDP gradient averaging over this sequence group.
        Loss normalization must use the global token/target count. Parameters
        must be replicated identically, in identical order, across the group.
        Every trainable parameter must participate; freeze unused parameters.
        This static contract avoids a device-to-host usage check each step
        and allows the entire forward/backward/reduction to be graph captured.
        """
        params = [p for p in module.parameters() if p.requires_grad]
        if self.world_size == 1 or not params:
            return
        if any(parameter.grad is None for parameter in params):
            raise ValueError("sequence gradient SUM requires a gradient for every trainable parameter; freeze unused parameters")
        buckets = {}
        for parameter in params:
            buckets.setdefault((parameter.device, parameter.dtype), []).append(parameter.grad)
        for gradients in buckets.values():
            flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
            self.sum_(flat)
            pieces = flat.split([g.numel() for g in gradients])
            torch._foreach_copy_(gradients, [piece.view_as(g) for g, piece in zip(gradients, pieces)])

    def halo(self, value, width=1):
        """Differentiable boundary exchange for a centered short convolution.

        Only boundary tokens are gathered. Interior tokens stay on their rank.
        The global ends are zero padded, matching native centered Conv1d.
        """
        if width < 1 or min(self.shard_lengths) < width:
            raise ValueError("halo width must fit every nonempty shard")
        if value.ndim != 3 or value.shape[1] != self.shard_lengths[self.rank]:
            raise ValueError("halo input must have shape [batch, local_length, channels]")
        zeros = value.new_zeros((value.shape[0], width, value.shape[-1]))
        if self.world_size == 1:
            return torch.cat((zeros, value, zeros), dim=1)
        from torch.distributed._functional_collectives import all_gather_single
        boundaries = torch.cat((value[:, :width], value[:, -width:]), dim=1).contiguous()
        gathered = all_gather_single(
            boundaries, 0, dist.group.WORLD if self.group is None else self.group,
        )
        neighbors = gathered.reshape(self.world_size, *boundaries.shape).unbind(0)
        left = neighbors[self.rank-1][:, width:] if self.rank else zeros
        right = neighbors[self.rank+1][:, :width] if self.rank+1 < self.world_size else zeros
        # Keep every rank in all_gather's backward, including unused endpoints.
        dependency = sum(neighbor[:, :0].sum() for neighbor in neighbors)
        return torch.cat((left, value, right), dim=1) + dependency
