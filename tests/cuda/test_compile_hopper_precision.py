"""Verification suite for fullgraph torch.compile, Hopper SM90a PTX, precision, and integration lifecycles."""
from __future__ import annotations

import copy
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from memsolve import CoreMode, MemSolve, MemSolveConfig
from memsolve.ball import cuda
from memsolve.ball.reference import rotary_position_factors


pytestmark = pytest.mark.cuda


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a64 = actual.detach().to(dtype=torch.float64)
    e64 = expected.detach().to(dtype=torch.float64)
    return float(
        torch.linalg.vector_norm(a64 - e64)
        / torch.linalg.vector_norm(e64).clamp_min(1e-12)
    )


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    try:
        cuda.load()
    except RuntimeError as error:
        pytest.skip(f"native CUDA extension is unavailable: {error}")


def _reference_deform_inputs(x: torch.Tensor):
    _, _, h, w = x.shape
    device = x.device

    def ref_points(shapes):
        pts = []
        for h_i, w_i in shapes:
            ry, rx = torch.meshgrid(
                torch.linspace(0.5, h_i - 0.5, h_i, dtype=torch.float32, device=device),
                torch.linspace(0.5, w_i - 0.5, w_i, dtype=torch.float32, device=device),
                indexing="ij",
            )
            pts.append(torch.stack((rx.reshape(-1)[None] / w_i, ry.reshape(-1)[None] / h_i), -1))
        return torch.cat(pts, 1)[:, :, None]

    s1 = torch.tensor([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], dtype=torch.long).to(device)
    i1 = torch.cat((s1.new_zeros((1,)), s1.prod(1).cumsum(0)[:-1]))
    r1 = ref_points([(h // 16, w // 16)])

    s2 = torch.tensor([(h // 16, w // 16)], dtype=torch.long).to(device)
    i2 = torch.cat((s2.new_zeros((1,)), s2.prod(1).cumsum(0)[:-1]))
    r2 = ref_points([(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)])
    return [r1, s1, i1], [r2, s2, i2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_vit_adapter_deform_inputs_cache_cudagraph_and_compile() -> None:
    pytest.importorskip("timm")
    from integrations import vit_adapter

    x = torch.empty(2, 3, 224, 256, device="cuda")
    vit_adapter._DEFORM_INPUTS_CACHE.clear()

    d1, d2 = vit_adapter.deform_inputs(x)
    ref1, ref2 = _reference_deform_inputs(x)
    for actual, expected in zip((*d1, *d2), (*ref1, *ref2)):
        if actual.is_floating_point():
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        else:
            assert torch.equal(actual, expected)

    # Persistent cache hit returns identical device storages without re-allocating.
    d1_hit, d2_hit = vit_adapter.deform_inputs(x)
    for a, b in zip((*d1, *d2), (*d1_hit, *d2_hit)):
        assert a.data_ptr() == b.data_ptr()

    # Cold CUDA Graph capture (empty cache) must succeed with zero pageable H2D copies,
    # and must NOT leak graph-private-pool tensors into the persistent cache.
    vit_adapter._DEFORM_INPUTS_CACHE.clear()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side_stream):
        with torch.cuda.graph(graph, stream=side_stream):
            g1, g2 = vit_adapter.deform_inputs(x)
    torch.cuda.current_stream().wait_stream(side_stream)
    assert (x.device, 224, 256) not in vit_adapter._DEFORM_INPUTS_CACHE
    graph.replay()
    for actual, expected in zip((*g1, *g2), (*ref1, *ref2)):
        if actual.is_floating_point():
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        else:
            assert torch.equal(actual, expected)

    # Dynamic fullgraph torch.compile across varying resolutions.
    compiled = torch.compile(vit_adapter.deform_inputs, fullgraph=True, dynamic=True)
    for height, width in ((192, 224), (256, 320)):
        probe = torch.empty(1, 3, height, width, device="cuda")
        c1, c2 = compiled(probe)
        e1, e2 = _reference_deform_inputs(probe)
        for actual, expected in zip((*c1, *c2), (*e1, *e2)):
            if actual.is_floating_point():
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            else:
                assert torch.equal(actual, expected)


def test_openmmlab_valid_mask_vectorized_and_thread_safe() -> None:
    pytest.importorskip("timm")
    from integrations import openmmlab

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.zeros(3, 3, 32, 40, device=device)
    samples = [
        SimpleNamespace(metainfo={"img_shape": (21, 37)}),
        SimpleNamespace(metainfo={"img_shape": (32, 40)}),
        SimpleNamespace(metainfo={"img_shape": (13, 19)}),
    ]
    actual = openmmlab._pixel_valid_mask(inputs, samples)
    expected = torch.zeros(3, 32, 40, dtype=torch.bool, device=device)
    expected[0, :21, :37] = True
    expected[1, :32, :40] = True
    expected[2, :13, :19] = True
    assert actual is not None and torch.equal(actual, expected)

    full_samples = [SimpleNamespace(metainfo={}) for _ in range(3)]
    full_mask = openmmlab._pixel_valid_mask(inputs, full_samples)
    assert full_mask is not None and bool(full_mask.all())

    single = openmmlab._pixel_valid_mask(inputs[:1], samples[:1])
    assert single is not None and torch.equal(single, expected[:1])

    with pytest.raises(ValueError, match="img_shape"):
        openmmlab._pixel_valid_mask(inputs[:1], [SimpleNamespace(metainfo={"img_shape": (0, 10)})])

    class _BaseModel:
        out_channels = 1
        test_cfg = SimpleNamespace(stride=(8, 8), crop_size=(16, 16))
        with_neck = False

        def backbone(self, batch_inputs: torch.Tensor, *, valid_mask: torch.Tensor | None = None):
            return (batch_inputs, valid_mask)

        def forward(self, inputs_: torch.Tensor, data_samples=None, mode: str = "tensor"):
            del data_samples, mode
            return self.extract_feat(inputs_)

    class _ConcurrentModel(openmmlab._SegmentationMaskMixin, openmmlab._MaskAwareMixin, _BaseModel):
        def encode_decode(self, crop: torch.Tensor, metas):
            del metas
            mask = self._memsolve_valid_mask
            assert mask is not None and mask.shape == crop[:, 0].shape
            return crop[:, :1].masked_fill(~mask[:, None], 0.0)

    shared = _ConcurrentModel()
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for step in range(20):
                h_valid = 10 + ((tid * 3 + step) % 20)
                w_valid = 12 + ((tid * 5 + step) % 24)
                batch = torch.ones(2, 3, 32, 40)
                meta = [
                    SimpleNamespace(metainfo={"img_shape": (h_valid, w_valid)}),
                    SimpleNamespace(metainfo={"img_shape": (32, w_valid)}),
                ]
                _, seen_mask = shared.forward(batch, meta)
                assert seen_mask is not None
                assert int(seen_mask[0].sum()) == h_valid * w_valid
                assert int(seen_mask[1].sum()) == 32 * w_valid
                assert shared._memsolve_valid_mask is None

                slide_mask = torch.zeros(1, 24, 24, dtype=torch.bool)
                slide_mask[:, : 8 + tid, : 9 + tid] = True
                shared._memsolve_valid_mask = slide_mask
                out = shared.slide_inference(torch.ones(1, 3, 24, 24), [{}])
                assert torch.equal(out[:, 0] > 0, slide_mask)
                assert shared._memsolve_valid_mask is slide_mask
                shared._memsolve_valid_mask = None
        except BaseException as exc:  # pragma: no cover - collected and re-raised
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]

    # Ephemeral instances dropped without clearing must release via WeakKeyDictionary.
    temp = _ConcurrentModel()
    temp_mask = torch.ones(1, 8, 8, dtype=torch.bool)
    mask_ref = weakref.ref(temp_mask)
    temp._memsolve_valid_mask = temp_mask
    del temp, temp_mask
    gc.collect()
    assert mask_ref() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_transformers_bert_attention_avoids_per_layer_cuda_load_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers", minversion="4.57.6")
    _ensure_cuda()
    from integrations import transformers as hf_integration

    cfg = hf_integration.MemSolveBertConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=128,
        memsolve_rank=16,
        memsolve_implementation="cuda",
        hidden_dropout_prob=0.0,
    )
    model = hf_integration.MemSolveBertModel(cfg).cuda().eval()
    ids = torch.randint(4, 64, (2, 32), device="cuda")
    mask = torch.ones_like(ids)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=ids, attention_mask=mask)

    load_calls = 0
    original_load = cuda.load

    def counting_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(cuda, "load", counting_load)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            out = model(input_ids=ids, attention_mask=mask)
            assert torch.isfinite(out.last_hidden_state).all()
    assert load_calls == 0


@pytest.mark.parametrize(
    "seq_len,rank,rotary,recompute",
    [
        (128, 32, False, False),
        (128, 32, True, True),
        (8, 32, True, False),
    ],
)
def test_memsolve_cuda_dynamo_explain_zero_graph_breaks(
    seq_len: int,
    rank: int,
    rotary: bool,
    recompute: bool,
) -> None:
    _ensure_cuda()
    torch._dynamo.reset()
    cfg = MemSolveConfig(
        dim=64,
        num_heads=2,
        rank=rank,
        core_mode=CoreMode.EQUILIBRIUM,
        rotary=rotary,
        recompute_readout=recompute,
    )
    layer = MemSolve(cfg).cuda()
    x = torch.randn(2, seq_len, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    valid_mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
    valid_mask[:, -2:] = False

    explanation = torch._dynamo.explain(layer)(
        x, valid_mask=valid_mask, implementation="cuda"
    )
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1


def test_vit_cpe_rmsnorm_2drove_block_fullgraph_dynamic_and_cudagraph() -> None:
    pytest.importorskip("timm")
    _ensure_cuda()
    import torch._functorch.config as functorch_config
    from integrations.timm import create_memsolve_vit

    functorch_config.backward_pass_autocast = "off"
    torch._dynamo.reset()
    torch.manual_seed(101)

    model_eager = create_memsolve_vit(
        image_size=224,
        patch_size=16,
        num_classes=10,
        embed_dim=128,
        depth=1,
        num_heads=4,
        rank=32,
        mlp_ratio=2.0,
        bias=True,
        implementation="cuda",
        drop_path_rate=0.0,
        dynamic_img_size=True,
        position_encoding="cpe",
        norm_type="rmsnorm",
        rotary_2d=True,
    ).cuda().eval()
    model_comp = copy.deepcopy(model_eager)
    eager_block = model_eager.blocks[0]
    compiled_block = torch.compile(
        model_comp.blocks[0],
        fullgraph=True,
        dynamic=True,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )

    for height, width in ((14, 14), (12, 16)):
        tokens = height * width
        imgs = torch.randn(2, 3, height * 16, width * 16, device="cuda")
        mask = torch.ones(2, tokens, device="cuda", dtype=torch.bool)
        mask[0, -5:] = False
        layout = model_eager._token_layout(imgs, mask)

        x_eager = torch.randn(2, tokens, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_comp = x_eager.detach().clone().requires_grad_()
        upstream = torch.randn(2, tokens, 128, device="cuda", dtype=torch.bfloat16)

        eager_block.zero_grad(set_to_none=True)
        model_comp.blocks[0].zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_eager = eager_block(x_eager, attn_mask=layout)
            y_comp = compiled_block(x_comp, attn_mask=layout)
        (y_eager.float() * upstream.float()).sum().backward()
        (y_comp.float() * upstream.float()).sum().backward()

        assert _relative_l2(y_comp, y_eager) <= 5e-3
        assert x_comp.grad is not None and x_eager.grad is not None
        assert _relative_l2(x_comp.grad, x_eager.grad) <= 1e-2
        for (name, p_e), (_, p_c) in zip(
            eager_block.named_parameters(), model_comp.blocks[0].named_parameters()
        ):
            assert p_c.grad is not None and p_e.grad is not None, name
            assert _relative_l2(p_c.grad, p_e.grad) <= 1e-2, name
    del y_eager, y_comp

    # Outer CUDAGraph capture & replay over the compiled ViT block.
    static_imgs = torch.randn(2, 3, 224, 224, device="cuda")
    static_mask = torch.ones(2, 196, device="cuda", dtype=torch.bool)
    static_mask[1, -8:] = False
    static_layout = model_eager._token_layout(static_imgs, static_mask)
    static_x = torch.randn(2, 196, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    static_up = torch.randn(2, 196, 128, device="cuda", dtype=torch.bfloat16)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            model_comp.blocks[0].zero_grad(set_to_none=False)
            static_x.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_warm = compiled_block(static_x, attn_mask=static_layout)
            (y_warm.float() * static_up.float()).sum().backward()
        del y_warm
        graph = torch.cuda.CUDAGraph()
        model_comp.blocks[0].zero_grad(set_to_none=False)
        static_x.grad = torch.zeros_like(static_x)
        with torch.cuda.graph(graph, stream=capture_stream):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y_graph = compiled_block(static_x, attn_mask=static_layout)
            loss_graph = (y_graph.float() * static_up.float()).sum()
            loss_graph.backward()
    torch.cuda.current_stream().wait_stream(capture_stream)

    for step in range(2):
        new_x = torch.randn_like(static_x)
        new_up = torch.randn_like(static_up)
        static_x.detach().copy_(new_x)
        static_up.copy_(new_up)
        for p in model_comp.blocks[0].parameters():
            if p.grad is not None:
                p.grad.zero_()
        static_x.grad.zero_()
        graph.replay()

        ref_x = new_x.detach().clone().requires_grad_()
        eager_block.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref_y = eager_block(ref_x, attn_mask=static_layout)
        (ref_y.float() * new_up.float()).sum().backward()

        assert _relative_l2(y_graph, ref_y) <= 5e-3, f"replay {step}"
        assert _relative_l2(static_x.grad, ref_x.grad) <= 1e-2, f"replay {step}"


class _StandaloneBertBlock(nn.Module):
    """BERT-style Post-LN block (centered short conv + MemSolve RoVE + SwiGLU) for any venv."""

    def __init__(self, dim: int = 128, heads: int = 4, rank: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.mixer = MemSolve(
            MemSolveConfig(
                dim=dim,
                num_heads=heads,
                rank=rank,
                core_mode=CoreMode.EQUILIBRIUM,
                bias=True,
                rotary=True,
                rotary_scaling="yarn",
                rotary_train_length=64,
            )
        )
        self.ln1 = nn.LayerNorm(dim, eps=1e-12)
        self.gate_up = nn.Linear(dim, 2 * 256)
        self.down = nn.Linear(256, dim)
        self.ln2 = nn.LayerNorm(dim, eps=1e-12)

    def forward(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden.masked_fill(~valid_mask[..., None], 0)
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x.masked_fill(~valid_mask[..., None], 0)
        if torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        mixed = self.mixer(
            x,
            valid_mask=valid_mask,
            implementation="cuda",
            rotary_factors=(cos, sin),
        )
        h = self.ln1(hidden + mixed)
        gate, up = self.gate_up(h).chunk(2, dim=-1)
        ffn = self.down(nn.functional.silu(gate) * up)
        return self.ln2(h + ffn)


def test_bert_shortconv_rove_block_fullgraph_dynamic_and_cudagraph() -> None:
    _ensure_cuda()
    import torch._functorch.config as functorch_config

    functorch_config.backward_pass_autocast = "off"
    torch._dynamo.reset()
    torch.manual_seed(202)

    eager = _StandaloneBertBlock(dim=128, heads=4, rank=32).cuda().eval()
    comp_mod = copy.deepcopy(eager)
    compiled = torch.compile(
        comp_mod,
        fullgraph=True,
        dynamic=True,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )

    for seq_len in (96, 144):
        pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(2, -1)
        cos, sin = rotary_position_factors(
            pos, head_dim=32, base=10000.0, train_length=64, context_length=seq_len, scaling="yarn"
        )
        mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
        mask[0, -9:] = False

        x_e = torch.randn(2, seq_len, 128, device="cuda", requires_grad=True)
        x_c = x_e.detach().clone().requires_grad_()
        up = torch.randn(2, seq_len, 128, device="cuda")

        eager.zero_grad(set_to_none=True)
        comp_mod.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_e = eager(x_e, mask, cos, sin)
            y_c = compiled(x_c, mask, cos, sin)
        (y_e.float() * up).sum().backward()
        (y_c.float() * up).sum().backward()

        assert _relative_l2(y_c, y_e) <= 5e-3
        assert _relative_l2(x_c.grad, x_e.grad) <= 1e-2

    # If HuggingFace transformers is installed in this venv, also verify real BertLayer fullgraph!
    try:
        from integrations.transformers import MemSolveBertConfig, MemSolveBertModel
    except ModuleNotFoundError:
        return

    cfg = MemSolveBertConfig(
        vocab_size=64,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=256,
        memsolve_rank=32,
        memsolve_short_conv_kernel=3,
        memsolve_implementation="cuda",
        hidden_dropout_prob=0.0,
    )
    hf_eager = MemSolveBertModel(cfg).cuda().eval().encoder.layer[0]
    hf_comp_mod = copy.deepcopy(hf_eager)
    hf_compiled = torch.compile(
        hf_comp_mod,
        fullgraph=True,
        dynamic=True,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )
    for seq_len in (80, 128):
        x_e = torch.randn(2, seq_len, 128, device="cuda", requires_grad=True)
        x_c = x_e.detach().clone().requires_grad_()
        up = torch.randn(2, seq_len, 128, device="cuda")
        mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
        mask[1, -6:] = False
        counts = mask.sum(-1, dtype=torch.float32).clamp_min(1.0)
        pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(2, -1)
        factors = hf_eager.attention.mixer._rotary_factors(x_e, mask, pos, None)
        shared = (mask, counts, factors)
        hf_eager.zero_grad(set_to_none=True)
        hf_comp_mod.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_e = hf_eager(x_e, cache_position=shared)[0]
            y_c = hf_compiled(x_c, cache_position=shared)[0]
        (y_e.float() * up).sum().backward()
        (y_c.float() * up).sum().backward()
        assert _relative_l2(y_c, y_e) <= 5e-3
        assert _relative_l2(x_c.grad, x_e.grad) <= 1e-2


def test_multiarch_native_solver_and_fake_tensor_symint() -> None:
    _ensure_cuda()
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    assert torch.ops.memsolve_equilibrium.contract_version() >= 17
    assert cuda.is_available()

    torch.manual_seed(303)
    for rank in (16, 32, 48, 64):
        eye = torch.eye(rank, device="cuda", dtype=torch.float32)
        raw = torch.randn(3, 2, rank, rank, device="cuda", dtype=torch.float32) * 0.2
        gram = raw @ raw.mT + 2.0 * eye
        factor = torch.ops.memsolve.compact_cholesky_inverse(gram)
        chol_ref = torch.linalg.solve_triangular(
            torch.linalg.cholesky(gram.double()),
            eye.double().expand_as(gram),
            upper=False,
        ).float()
        assert _relative_l2(factor, chol_ref) <= 1e-5

        system = eye + torch.randn(3, 2, rank, rank, device="cuda", dtype=torch.float32) * 0.1
        rhs = torch.randn(3, 2, rank, 40, device="cuda", dtype=torch.float32)
        lu_sol = torch.ops.memsolve.compact_lu_solve(system, rhs)
        lu_ref = torch.linalg.solve(system.double(), rhs.double()).float()
        assert _relative_l2(lu_sol, lu_ref) <= 1e-5

    # Dynamic SymInt FakeTensorMode trace across native ops and full MemSolve fast path.
    from torch._dynamo.source import LocalSource

    real_layer = MemSolve(
        MemSolveConfig(dim=64, num_heads=2, rank=32, rotary=True, recompute_readout=True)
    ).cuda()
    shape_env = ShapeEnv()
    sym_b = shape_env.create_symintnode(shape_env.create_symbol(3, LocalSource("b")), hint=3)
    sym_n = shape_env.create_symintnode(shape_env.create_symbol(97, LocalSource("n")), hint=97)
    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True):
        b_sym = torch.empty(
            (sym_b, sym_n, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        assert isinstance(b_sym.shape[0], torch.SymInt)
        assert isinstance(b_sym.shape[1], torch.SymInt)
        mask_sym = torch.ones(b_sym.shape[:2], device="cuda", dtype=torch.bool)
        out_sym = real_layer(b_sym, valid_mask=mask_sym, implementation="cuda")
        assert out_sym.shape == b_sym.shape
        assert out_sym.dtype == torch.bfloat16
        out_sym.float().sum().backward()
        assert b_sym.grad is not None and b_sym.grad.shape == b_sym.shape

        # Also check low-level native ops with symbolic batch size.
        sym_gram = torch.empty(
            (b_sym.shape[0], 2, 32, 32), device="cuda", dtype=torch.float32
        )
        sym_rhs = torch.empty(
            (b_sym.shape[0], 2, 32, 48), device="cuda", dtype=torch.float32
        )
        inv = torch.ops.memsolve.compact_cholesky_inverse(sym_gram)
        sol = torch.ops.memsolve.compact_lu_solve(sym_gram, sym_rhs)
        assert inv.shape == sym_gram.shape
        assert sol.shape == sym_rhs.shape


def test_hopper_sm90a_offline_compile_zero_spills_and_ptx() -> None:
    from triton.backends.compiler import GPUTarget

    target = GPUTarget("cuda", 90, 32)
    compiled = cuda._offline_compile_hopper_check(target)
    expected_kernels = {
        "biased_gemm",
        "statistics_wgmma",
        "chunk_statistics_wgmma",
        "readout",
        "token_backward",
    }
    assert expected_kernels.issubset(compiled.keys())
    for name, kernel in compiled.items():
        assert getattr(kernel, "n_spills", 0) == 0, name
        ptx = kernel.asm["ptx"]
        assert "cp.async.bulk.tensor" in ptx, name
        assert "wgmma.mma_async" in ptx, name
        assert "griddepcontrol.wait" in ptx and "griddepcontrol.launch_dependents" in ptx, name
        assert "sqrt.approx.f32" not in ptx and "div.approx.f32" not in ptx, name
        assert ".local" not in ptx, name
    assert "wgmma.mma_async.sync.aligned.m64n32k16" in compiled["statistics_wgmma"].asm["ptx"]
    assert "wgmma.mma_async.sync.aligned.m64n32k16" in compiled["chunk_statistics_wgmma"].asm["ptx"]


@pytest.mark.parametrize(
    "regime,seq_len,rank,scale,out_tol,grad_tol",
    [
        ("standard", 256, 32, 1.0, 4e-3, 1.2e-2),
        ("short_lt_rank", 8, 32, 1.0, 4e-3, 1.2e-2),
        ("short_eq_rank", 32, 32, 1.0, 4e-3, 1.2e-2),
        ("strong_relation_r32", 256, 32, 64.0, 1.5e-2, 3.5e-2),
        ("strong_relation_r64", 256, 64, 64.0, 1.5e-2, 3.5e-2),
    ],
)
def test_cuda_fast_path_precision_against_fp64_oracle(
    regime: str,
    seq_len: int,
    rank: int,
    scale: float,
    out_tol: float,
    grad_tol: float,
) -> None:
    _ensure_cuda()
    torch.manual_seed(404 + seq_len + rank + int(scale))

    cfg = MemSolveConfig(
        dim=128,
        num_heads=2,
        rank=rank,
        core_mode=CoreMode.EQUILIBRIUM,
        bias=True,
        rotary=True,
    )
    fast = MemSolve(cfg).cuda()
    with torch.no_grad():
        fast.core_base_raw.normal_(std=0.15)
        if scale != 1.0:
            # Amplify relation projection rows [0 : H*r] while keeping content bounded.
            fast.w_bc.weight[: 2 * rank].mul_(scale)
            if fast.w_bc.bias is not None:
                fast.w_bc.bias[: 2 * rank].mul_(scale * 0.25)

    oracle = copy.deepcopy(fast).to(dtype=torch.float64)

    x_bf16 = torch.randn(2, seq_len, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_fp64 = x_bf16.detach().to(dtype=torch.float64).requires_grad_()
    valid_mask = torch.ones(2, seq_len, device="cuda", dtype=torch.bool)
    if seq_len > 4:
        valid_mask[0, -2:] = False
    upstream_bf16 = torch.randn(2, seq_len, 128, device="cuda", dtype=torch.bfloat16)
    upstream_fp64 = upstream_bf16.to(dtype=torch.float64)

    y_fast = fast(x_bf16, valid_mask=valid_mask, implementation="cuda")
    y_oracle = oracle(x_fp64, valid_mask=valid_mask, implementation="reference")

    out_err = _relative_l2(y_fast, y_oracle)
    assert torch.isfinite(y_fast).all(), regime
    assert out_err <= out_tol, f"{regime}: output rel_l2={out_err:.6f} > {out_tol}"

    loss_fast = (y_fast.double() * upstream_fp64).sum()
    loss_oracle = (y_oracle * upstream_fp64).sum()

    if regime == "short_lt_rank":
        # Verify analytical short-N backward supports retain_graph=True cleanly.
        g_first = torch.autograd.grad(
            loss_fast, (x_bf16, *fast.parameters()), retain_graph=True
        )
        g_fast = torch.autograd.grad(
            loss_fast, (x_bf16, *fast.parameters()), retain_graph=False
        )
        for a, b in zip(g_first, g_fast):
            assert torch.equal(a, b)
    else:
        g_fast = torch.autograd.grad(loss_fast, (x_bf16, *fast.parameters()))
    g_oracle = torch.autograd.grad(loss_oracle, (x_fp64, *oracle.parameters()))

    names = ("x", *[name for name, _ in fast.named_parameters()])
    for name, gf, go in zip(names, g_fast, g_oracle):
        assert torch.isfinite(gf).all(), f"{regime}:{name} non-finite"
        err = _relative_l2(gf, go)
        assert err <= grad_tol, f"{regime}:{name} grad rel_l2={err:.6f} > {grad_tol}"


def test_compiled_custom_ops_zero_dead_buffers_single_weight_cast_and_dynamic_auto() -> None:
    _ensure_cuda()
    from memsolve.ball.reference import tensor_core_linear

    torch._dynamo.reset()
    torch.manual_seed(505)

    # 1. Static compile at N=196 > r=32 with recompute_readout="auto":
    #    - resolves "auto" to False (N <= 512), bypassing saved_mixed completely
    #    - allocates 0-element tok_inv_buf and res_buf (zero dead buffers)
    #    - saves BF16 weights instead of FP32 weights (single forward cast)
    cfg = MemSolveConfig(
        dim=192,
        num_heads=3,
        rank=32,
        core_mode=CoreMode.EQUILIBRIUM,
        bias=True,
        rotary=True,
        recompute_readout="auto",
    )
    eager_layer = MemSolve(cfg).cuda()
    comp_layer = copy.deepcopy(eager_layer)
    compiled_static = torch.compile(comp_layer, fullgraph=True, dynamic=False)

    x_s = torch.randn(2, 196, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y_warm = compiled_static(x_s, implementation="cuda")
    y_warm.float().sum().backward()
    comp_layer.zero_grad(set_to_none=True)
    x_s.grad = None

    saved_specs: list[tuple[tuple[int, ...], torch.dtype, int]] = []

    def pack_hook(t: torch.Tensor) -> torch.Tensor:
        saved_specs.append((tuple(t.shape), t.dtype, t.untyped_storage().nbytes()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        y_c = compiled_static(x_s, implementation="cuda")
    # Neither tok_inv_buf [2, 3, 32, 32] (beyond inv_buf) nor res_buf [2, 3, 32, 64]
    # (beyond coef_buf + state_buf) nor saved_mixed [2*512, 192] nor FP32 linear weights
    # ([288, 192] / [192, 192]) may be saved for backward.
    fp32_rank_squares = [s for s in saved_specs if s[0] == (2, 3, 32, 32) and s[1] == torch.float32]
    assert len(fp32_rank_squares) == 1, f"expected only inv_buf (not tok_inv_buf): {saved_specs}"
    fp32_rank_states = [s for s in saved_specs if s[0] == (2, 3, 32, 64) and s[1] == torch.float32]
    assert len(fp32_rank_states) == 2, f"expected only coef_buf+state_buf (not res_buf): {saved_specs}"
    assert not any(s[0] == (2 * 512, 192) for s in saved_specs), saved_specs
    assert not any(s[0] in ((288, 192), (192, 192)) and s[1] == torch.float32 for s in saved_specs), saved_specs
    assert any(s[0] == (288, 192) and s[1] == torch.bfloat16 for s in saved_specs), saved_specs
    assert any(s[0] == (192, 192) and s[1] == torch.bfloat16 for s in saved_specs), saved_specs
    y_c.float().sum().backward()

    # 2. Verify FP32 & BF16 weight gradient parity in compiled tensor_core_linear.
    for w_dtype in (torch.float32, torch.bfloat16):
        w_e = torch.randn(96, 64, device="cuda", dtype=w_dtype, requires_grad=True)
        b_e = torch.randn(96, device="cuda", dtype=w_dtype, requires_grad=True)
        v_e = torch.randn(3, 40, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w_c = w_e.detach().clone().requires_grad_()
        b_c = b_e.detach().clone().requires_grad_()
        v_c = v_e.detach().clone().requires_grad_()
        up = torch.randn(3, 40, 96, device="cuda", dtype=torch.bfloat16)

        fn_comp = torch.compile(
            lambda v, w, b: tensor_core_linear(
                v, w, b, output_dtype=torch.bfloat16, fuse_weight_bias_grad=True
            ),
            fullgraph=True,
        )
        out_e = tensor_core_linear(v_e, w_e, b_e, output_dtype=torch.bfloat16, fuse_weight_bias_grad=True)
        out_c = fn_comp(v_c, w_c, b_c)
        (out_e.float() * up.float()).sum().backward()
        (out_c.float() * up.float()).sum().backward()
        assert out_c.dtype == torch.bfloat16
        assert w_c.grad is not None and w_c.grad.dtype == w_dtype
        assert b_c.grad is not None and b_c.grad.dtype == w_dtype
        assert _relative_l2(out_c, out_e) <= 1e-5
        assert _relative_l2(v_c.grad, v_e.grad) <= 1e-5
        assert _relative_l2(w_c.grad, w_e.grad) <= 1e-4
        assert _relative_l2(b_c.grad, b_e.grad) <= 1e-4

    # 3. Dynamic SymInt compile with recompute_readout="auto":
    #    - (B=4, N=256): B*N = 1024 > 512 while N = 256 <= 512 -> bounds-safe retained mixed!
    #    - (B=2, N=640): N = 640 > 512 -> recomputes readout within the same dynamic graph!
    dyn_eager = MemSolve(cfg).cuda()
    dyn_comp_mod = copy.deepcopy(dyn_eager)
    compiled_dyn = torch.compile(dyn_comp_mod, fullgraph=True, dynamic=True)

    for batch_sz, seq_len in ((4, 256), (2, 640)):
        xe = torch.randn(batch_sz, seq_len, cfg.dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        xc = xe.detach().clone().requires_grad_()
        up = torch.randn(batch_sz, seq_len, cfg.dim, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(batch_sz, seq_len, device="cuda", dtype=torch.bool)
        mask[0, -3:] = False

        dyn_eager.zero_grad(set_to_none=True)
        dyn_comp_mod.zero_grad(set_to_none=True)
        ye = dyn_eager(xe, valid_mask=mask, implementation="cuda")
        yc = compiled_dyn(xc, valid_mask=mask, implementation="cuda")
        (ye.float() * up.float()).sum().backward()
        (yc.float() * up.float()).sum().backward()

        assert _relative_l2(yc, ye) <= 5e-3
        assert _relative_l2(xc.grad, xe.grad) <= 1e-2
        for (name, pe), (_, pc) in zip(dyn_eager.named_parameters(), dyn_comp_mod.named_parameters()):
            assert pc.grad is not None and pe.grad is not None, name
            assert _relative_l2(pc.grad, pe.grad) <= 1e-2, name


def test_all_sm80_triton_specializations_zero_spills() -> None:
    _ensure_cuda()
    torch.manual_seed(606)

    # Sweep ranks x head_dims x rotary x core_mode x sequence regimes (>= 84 cubin specializations)
    # including r > 32 + RoVE + wide head_dim + odd unaligned head_dim + chunked stats + reduction.
    cases: list[tuple[int, int, bool, CoreMode, int]] = []
    for rank in (16, 32, 48, 64):
        for head_dim in (16, 32, 48, 64, 96, 128):
            for rotary in (False, True):
                cases.append((rank, head_dim, rotary, CoreMode.EQUILIBRIUM, 196))
        # Unaligned odd head_dim (non-rotary), complement mode, short-N, chunked stats, and tiles >= 32
        cases.append((rank, 17, False, CoreMode.EQUILIBRIUM, 129))
        cases.append((rank, 64, True, CoreMode.COMPLEMENT, 196))
        cases.append((rank, 64, True, CoreMode.EQUILIBRIUM, 8))
        cases.append((rank, 64, True, CoreMode.EQUILIBRIUM, 2048))
    cases.append((32, 64, True, CoreMode.EQUILIBRIUM, 16384))

    for rank, head_dim, rotary, mode, seq_len in cases:
        heads = 2
        dim = heads * head_dim
        layer = MemSolve(
            MemSolveConfig(
                dim=dim,
                num_heads=heads,
                rank=rank,
                core_mode=mode,
                bias=True,
                rotary=rotary,
                recompute_readout=False,
            )
        ).cuda()
        x = torch.randn(1, seq_len, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = layer(x, implementation="cuda")
        out.float().sum().backward()

    triton_kernels = [
        cuda._biased_gemm_kernel(),
        *cuda._no_frame_kernels(),
        *cuda._stats_reduction_kernels(),
        *cuda._compact_kernels(),
        *cuda._compact_adjoint_kernels(),
        *cuda._short_kernels(),
    ]
    total_cubins = 0
    spilled: list[tuple[str, int]] = []
    for kernel in triton_kernels:
        kname = getattr(kernel, "__name__", str(kernel))
        for cache in kernel.device_caches.values():
            for compiled in cache[0].values():
                total_cubins += 1
                spills = getattr(compiled, "n_spills", 0)
                if spills != 0:
                    spilled.append((kname, spills))

    assert total_cubins >= 84, f"expected >= 84 compiled Triton cubins, got {total_cubins}"
    assert not spilled, f"Triton cubin register spills detected: {spilled}"

