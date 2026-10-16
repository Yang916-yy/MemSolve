"""Masked LRA attention adapters with optional value-path RoVE.

Algorithms follow Performer, Nyströmformer, and the pure Agent/MiTA/PoLA
operators used by MiTA. See THIRD_PARTY_NOTICES.md for pinned sources.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from memsolve.ball.reference import rotate_content

LRA_ATTENTION = frozenset({'performer', 'nystromformer', 'agent', 'mita', 'polaformer', 'local', 'linear', 'bigbird'})


def masked_softmax(scores, mask):
    scores = scores.masked_fill(~mask, -torch.inf)
    # Empty rows have zero mass, including their backward pass.
    present = mask.any(dim=-1, keepdim=True)
    scores = torch.where(present, scores, torch.zeros_like(scores))
    return scores.softmax(-1).masked_fill(~mask, 0)


def pooled_landmarks(x, valid, count, grid=None):
    """Mask-aware contiguous pooling; image routers use rectangular spatial bins."""
    b, h, n, d = x.shape
    dtype = x.dtype
    x = x.float() if dtype in (torch.bfloat16, torch.float16) else x
    count = min(count, n)
    if grid is None:
        bins = torch.arange(n, device=x.device) * count // n
    else:
        rows, cols = grid
        pr = min(rows, math.isqrt(count))
        while count % pr:
            pr -= 1
        pc = count // pr
        positions = torch.arange(n, device=x.device)
        bins = (positions // cols * pr // rows) * pc + positions % cols * pc // cols
    weights = valid[:, None, :, None].to(x.dtype)
    index = bins[None, None, :, None]
    sums = x.new_zeros(b, h, count, d).scatter_add(2, index.expand(b, h, n, d), x * weights)
    counts = x.new_zeros(b, 1, count, 1).scatter_add(2, index.expand(b, 1, n, 1), weights)
    return (sums / counts.clamp_min(1)).to(dtype), counts[:, 0, :, 0] > 0


def iterative_inverse(matrix, iterations=6):
    """Nyströmformer's sixth-order Newton–Schulz iteration."""
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    norm1 = matrix.abs().sum(-2).amax(-1)
    norminf = matrix.abs().sum(-1).amax(-1)
    inverse = matrix.transpose(-1, -2) / (norm1 * norminf).clamp_min(1e-12)[..., None, None]
    for _ in range(iterations):
        product = matrix @ inverse
        inverse = .25 * inverse @ (13 * identity - product @ (15 * identity - product @ (7 * identity - product)))
    return inverse


def orthogonal_features(features, dim):
    blocks = [torch.linalg.qr(torch.randn(dim, dim), mode='reduced').Q.T
              for _ in range(math.ceil(features / dim))]
    directions = torch.cat(blocks)[:features]
    return torch.randn(features, dim).norm(dim=-1)[:, None] * directions


@torch.compiler.disable
def routed_attention(q, k, v, routers, router_valid, valid, topk):
    """MiTA's joint agent/routed softmax using public variable-length FlashAttention.

    Concatenating the two memories is equivalent to upstream's LSE merge.
    Routing changes segment lengths, so this operation stays outside CUDA Graph.
    """
    b, h, n, d = q.shape
    m = routers.shape[2]
    topk = min(topk, n)
    # Routing scores do not need an autograd graph: only the selected indices
    # enter the differentiable attention calculation.
    with torch.no_grad():
        scores = routers.float() @ k.float().transpose(-1, -2)
        chosen = scores.masked_fill(~valid[:, None, None, :], -torch.inf).topk(topk, dim=-1).indices
        gate = routers.float() @ q.float().transpose(-1, -2)
        expert = gate.masked_fill(~router_valid[:, None, :, None], -torch.inf).argmax(-2)
    agent_values = F.scaled_dot_product_attention(
        routers, k, v, attn_mask=valid[:, None, None, :])
    bh = b * h
    query_valid = valid[:, None].expand(b, h, n).reshape(-1)
    positions = torch.nonzero(query_valid, as_tuple=True)[0]
    if positions.numel() == 0:
        return (q + k + v + routers.sum() * 0) * 0
    groups = (expert + torch.arange(bh, device=q.device).reshape(b, h, 1) * m).flatten()[positions]
    order = groups.argsort()
    positions = positions[order]
    counts = torch.bincount(groups, minlength=bh * m)
    nonempty = torch.nonzero(counts, as_tuple=True)[0]
    group_bh = nonempty // m
    raw_indices = chosen.reshape(bh * m, topk)[nonempty]
    raw_indices = raw_indices + group_bh[:, None] * n
    raw_k = k.reshape(bh * n, d)[raw_indices]
    raw_v = v.reshape(bh * n, d)[raw_indices]
    all_valid = valid[:, None].expand(b, h, n).reshape(-1)
    raw_valid = all_valid[raw_indices]
    # Upstream MiTA smooths agent keys by an additional 1/sqrt(d).
    agent_k = routers.reshape(bh, m, d)[group_bh] * d ** -.5
    agent_v = agent_values.reshape(bh, m, d)[group_bh]
    agent_valid = router_valid[:, None].expand(b, h, m).reshape(bh, m)[group_bh]
    memory_valid = torch.cat((agent_valid, raw_valid), dim=1)
    memory_k = torch.cat((agent_k, raw_k), dim=1)
    memory_v = torch.cat((agent_v, raw_v), dim=1)
    queries = q.reshape(bh * n, d)[positions]
    qcounts = counts[nonempty]
    if q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
        from torch.nn.attention.varlen import varlen_attn
        kcounts = memory_valid.sum(-1)
        cuq = F.pad(qcounts.cumsum(0), (1, 0)).to(torch.int32)
        cuk = F.pad(kcounts.cumsum(0), (1, 0)).to(torch.int32)
        output = varlen_attn(queries[:, None].contiguous(),
                            memory_k[memory_valid][:, None].contiguous(),
                            memory_v[memory_valid][:, None].contiguous(),
                            cuq, cuk, int(qcounts.max()), m + topk)[:, 0]
    else:
        chunks = queries.split(qcounts.tolist())
        output = torch.cat([
            F.scaled_dot_product_attention(chunk[None, None], memory_k[i][None, None],
                memory_v[i][None, None], attn_mask=memory_valid[i][None, None, None])[0, 0]
            for i, chunk in enumerate(chunks)
        ])
    return q.new_zeros(bh * n, d).index_copy(0, positions, output).reshape(b, h, n, d)


class LRAAttention(nn.Module):
    def __init__(self, dim, heads, kind, *, bias=True, landmarks=128, features=128,
                 routed_keys=128, grid_shape=None):
        super().__init__()
        if kind not in LRA_ATTENTION or dim % heads:
            raise ValueError('unsupported LRA attention or head dimension')
        self.kind, self.heads, self.head_dim = kind, heads, dim // heads
        self.landmarks, self.features, self.routed_keys = landmarks, features, routed_keys
        self.grid_shape = grid_shape
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.local_block_mask = None
        self.local_flex = None
        if kind == 'performer':
            self.register_buffer('projection', orthogonal_features(features, self.head_dim))
        if kind == 'polaformer':
            if self.head_dim % 2:
                raise ValueError('PoLA splits each value head into two equal parts')
            self.power = nn.Parameter(torch.zeros(1, heads, 1, self.head_dim))
            self.scale = nn.Parameter(torch.zeros(1, heads, 1, self.head_dim))

    def prepare_local(self, length, device):
        """Cache the position-only sparse mask before compile / CUDA Graph capture."""
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        # 256 keys: 128 to the left, self, and 127 to the right.
        def window(b, h, q, k):
            return (q - k <= 128) & (k - q < 128)
        self.local_block_mask = create_block_mask(
            window, None, None, length, length, device=str(device))
        self.local_flex = torch.compile(flex_attention, dynamic=False, fullgraph=True)

    def prepare_bigbird(self, length, device, seed=0):
        """Use Hugging Face's BigBird random plan with a compiled sparse kernel."""
        import numpy as np
        from transformers import BigBirdConfig
        from transformers.models.big_bird.modeling_big_bird import BigBirdBlockSparseAttention
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        if length % 64 or length < 1024:
            raise ValueError("BigBird baseline requires a multiple of 64 and at least 1024 tokens")
        config = BigBirdConfig(hidden_size=self.heads * self.head_dim,
            num_attention_heads=self.heads, block_size=64, num_random_blocks=3,
            max_position_embeddings=max(4096, length))
        # Planning must not consume the training RNG stream.
        numpy_state = np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            upstream = BigBirdBlockSparseAttention(config, seed=seed).train()
        try:
            np.random.seed(seed)
            if length in (1024, 3072, 4096):
                random = np.stack([upstream._bigbird_block_rand_mask(
                    config.max_position_embeddings, config.max_position_embeddings,
                    64, 64, 3, last_idx=1024)[:length // 64 - 2]
                    for _ in range(self.heads)])
            else:
                plan, counts = upstream._get_rand_attn_plan(length, 64, 3)
                random = np.stack(upstream._bigbird_block_rand_mask_with_head(
                    length, length, 64, 64, self.heads, plan, counts))
        finally:
            np.random.set_state(numpy_state)
        blocks = length // 64
        multiplicity = torch.zeros(self.heads, blocks, blocks)
        multiplicity[:, 0, :] = multiplicity[:, -1, :] = 1
        for h in range(self.heads):
            for q in range(1, blocks - 1):
                keys = sorted({0, blocks - 1, q - 1, q, q + 1})
                multiplicity[h, q, keys] = 1
                for k in random[h, q - 1]:
                    multiplicity[h, q, int(k)] += 1
        self.register_buffer('bigbird_counts', multiplicity.to(device), persistent=True)
        def pattern(b, h, q, k):
            return self.bigbird_counts[h, q // 64, k // 64] > 0
        self.local_block_mask = create_block_mask(
            pattern, None, self.heads, length, length, device=str(device), BLOCK_SIZE=64)
        self.local_flex = torch.compile(flex_attention, dynamic=False, fullgraph=True)

    def _bigbird(self, q, k, v, valid):
        def padding(score, b, h, qi, ki):
            count = self.bigbird_counts[h, qi // 64, ki // 64]
            return torch.where(valid[b, ki], score + count.clamp_min(1).log(), -float('inf'))
        return self.local_flex(q, k, v, score_mod=padding, block_mask=self.local_block_mask)

    def _local(self, q, k, v, valid):
        if self.local_flex is not None:
            def padding(score, b, h, qi, ki):
                return torch.where(valid[b, ki], score, -float('inf'))
            return self.local_flex(q, k, v, score_mod=padding,
                                   block_mask=self.local_block_mask)
        positions = torch.arange(q.shape[-2], device=q.device)
        distance = positions[:, None] - positions[None, :]
        mask = (distance <= 128) & (distance > -128)
        return F.scaled_dot_product_attention(q, k, v,
            attn_mask=mask[None, None] & valid[:, None, None])

    def _linear(self, q, k, v, valid):
        """Idiap fast-transformers: ELU+1 features, KV reduction, epsilon denominator."""
        q, k = F.elu(q) + 1, F.elu(k) + 1
        k = k.masked_fill(~valid[:, None, :, None], 0)
        kv = k.transpose(-1, -2) @ v
        denominator = (q * k.sum(-2, keepdim=True)).sum(-1, keepdim=True) + 1e-6
        return (q @ kv) / denominator

    def _performer(self, q, k, v, valid):
        d = self.head_dim
        q, k = q * d ** -.25, k * d ** -.25
        qproj, kproj = q @ self.projection.to(q.dtype).T, k @ self.projection.to(k.dtype).T
        qshift = qproj.amax(-1, keepdim=True).detach()
        kshift = kproj.masked_fill(~valid[:, None, :, None], -torch.inf).amax((-2, -1), keepdim=True)
        kshift = torch.where(valid.any(-1)[:, None, None, None], kshift, torch.zeros_like(kshift)).detach()
        qp = ((qproj - q.square().sum(-1, keepdim=True) * .5 - qshift).exp() + 1e-4) * self.features ** -.5
        kp = ((kproj - k.square().sum(-1, keepdim=True) * .5 - kshift).exp() + 1e-4) * self.features ** -.5
        kp = kp.masked_fill(~valid[:, None, :, None], 0)
        normalizer = (qp * kp.sum(-2, keepdim=True)).sum(-1, keepdim=True)
        return (qp @ (kp.transpose(-1, -2) @ v)) / normalizer.clamp_min(1e-6)

    def _nystrom(self, q, k, v, valid):
        q, k = q * self.head_dim ** -.25, k * self.head_dim ** -.25
        qland, landvalid = pooled_landmarks(q, valid, self.landmarks)
        kland, _ = pooled_landmarks(k, valid, self.landmarks)
        a = masked_softmax(q @ kland.transpose(-1, -2), landvalid[:, None, None])
        b = masked_softmax(qland @ kland.transpose(-1, -2), landvalid[:, None, None])
        b = b.masked_fill(~landvalid[:, None, :, None], 0)
        c = masked_softmax(qland @ k.transpose(-1, -2), valid[:, None, None])
        c = c.masked_fill(~landvalid[:, None, :, None], 0)
        return a @ (iterative_inverse(b) @ (c @ v))

    def _pola(self, q, k, v, valid):
        q, k = q / F.softplus(self.scale), k / F.softplus(self.scale)
        power = 1 + 4 * self.power.sigmoid()
        qp, qn = q.relu().pow(power), (-q).relu().pow(power)
        kp, kn = k.relu().pow(power), (-k).relu().pow(power)
        sim, opp, keys = torch.cat((qp, qn), -1), torch.cat((qn, qp), -1), torch.cat((kp, kn), -1)
        keys = keys.masked_fill(~valid[:, None, :, None], 0)
        count = valid.sum(-1)[:, None, None, None].to(q.dtype).clamp_min(1)
        mean = keys.sum(-2, keepdim=True) / count
        v1, v2 = v.chunk(2, dim=-1)
        def branch(query, value):
            denominator = (query * mean).sum(-1, keepdim=True) + 1e-6
            return query @ ((keys.transpose(-1, -2) * count.rsqrt()) @ (value * count.rsqrt())) / denominator
        return torch.cat((branch(sim, v1), branch(opp, v2)), -1)

    def forward(self, x, valid_mask, rotary_factors=None):
        b, n, dim = x.shape
        x = x.masked_fill(~valid_mask[:, :, None], 0)
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        dtype = v.dtype
        if rotary_factors is not None:
            v = rotate_content(v, *rotary_factors).to(dtype)
        # Sensitive feature maps, denominators and iterative inverses accumulate
        # in FP32; FlashAttention handles softmax accumulation internally.
        if self.kind in ('performer', 'nystromformer', 'polaformer', 'linear'):
            with torch.autocast(device_type=x.device.type, enabled=False):
                calc = torch.float64 if q.dtype == torch.float64 else torch.float32
                q, k, v = q.to(calc), k.to(calc), v.to(calc)
                fn = {'performer': self._performer, 'nystromformer': self._nystrom, 'polaformer': self._pola, 'linear': self._linear}[self.kind]
                y = fn(q, k, v, valid_mask).to(dtype)
        elif self.kind == 'bigbird':
            y = self._bigbird(q, k, v, valid_mask)
        elif self.kind == 'local':
            y = self._local(q, k, v, valid_mask)
        else:
            routers, router_valid = pooled_landmarks(q, valid_mask, self.landmarks, self.grid_shape)
            if self.kind == 'mita':
                y = routed_attention(q, k, v, routers, router_valid, valid_mask, self.routed_keys)
            else:
                agent_v = F.scaled_dot_product_attention(routers, k, v, attn_mask=valid_mask[:, None, None])
                y = F.scaled_dot_product_attention(q, routers, agent_v, attn_mask=router_valid[:, None, None])
        if rotary_factors is not None:
            y = rotate_content(y, *rotary_factors, inverse=True).to(dtype)
        y = self.proj(y.transpose(1, 2).reshape(b, n, dim))
        return y.masked_fill(~valid_mask[:, :, None], 0)
