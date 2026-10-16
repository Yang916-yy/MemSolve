"""Two-rank exact operator and centered-convolution gradient contracts."""
from __future__ import annotations

import copy
from datetime import timedelta
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.cuda


def _relative(actual, expected):
    return float(((actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-12)).detach())


def _worker(rank, rendezvous):
    from memsolve import MemSolve, MemSolveConfig, CoreMode
    from memsolve.ball import cuda
    from integrations.distributed import SequenceParallel
    from integrations.transformers import _CenteredShortConv
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=180))
    graph = None
    try:
        cuda.load()
        for operator_rank, lengths in [(16,(97,100)), (32,(19,22)), (48,(97,100)), (64,(97,100))]:
            context = SequenceParallel.create(lengths[rank])
            for mode in CoreMode:
                for masked in (False, True, "remote"):
                    torch.manual_seed(913+operator_rank)
                    model = MemSolve(MemSolveConfig(128,2,rank=operator_rank,bias=True,core_mode=mode)).cuda()
                    if model.core_base_raw is not None:
                        with torch.no_grad(): model.core_base_raw.normal_(std=.1)
                    # Generate common inputs on CPU; separate GPU RNG state is irrelevant.
                    full = torch.randn(2,sum(lengths),128,dtype=torch.bfloat16).cuda()
                    upstream = torch.randn(full.shape,dtype=torch.bfloat16).cuda()*.01
                    mask = torch.ones(2,sum(lengths),dtype=torch.bool,device="cuda")
                    if masked:
                        mask[0,::3]=False; mask[1]=False
                        if masked == "remote":mask[:,:lengths[0]]=True
                        full[~mask]=float('nan')
                    selection = slice(context.offset,context.offset+lengths[rank])
                    x = full[:,selection].clone().requires_grad_()
                    local_mask = mask[:,selection] if masked and not (masked == "remote" and rank == 0) else None
                    counts = context.valid_counts(local_mask,batch_size=2,device=x.device)
                    output = model(x,local_mask,implementation="cuda",sequence_parallel=context,valid_counts=counts)
                    (output.float()*upstream[:,selection].float()).sum().backward()
                    context.reduce_gradients_(model)
                    reference = copy.deepcopy(model).double()
                    reference.zero_grad(set_to_none=True)
                    rx=full.double().requires_grad_()
                    expected=reference(rx,mask if masked else None,implementation="reference")
                    (expected*upstream.double()).sum().backward()
                    assert _relative(output,expected[:,selection]) < .005
                    assert _relative(x.grad,rx.grad[:,selection]) < .03
                    for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
                        assert _relative(p.grad,q.grad)<.03,(operator_rank,mode,name,_relative(p.grad,q.grad))
                    if masked is True:
                        assert torch.count_nonzero(output[1])==0
                    if local_mask is not None:
                        assert torch.count_nonzero(x.grad[~local_mask])==0
        # Capture compact collectives and the final parameter SUM together.
        # Replay reads changed inputs and weights. Compare to unsharded CUDA
        # to isolate capture/reduction; independent FP64 oracles are above.
        context=SequenceParallel.create(65)
        torch.manual_seed(417)
        model=MemSolve(MemSolveConfig(64,1,rank=32,bias=True)).cuda()
        full=torch.randn(1,130,64,dtype=torch.bfloat16).cuda()
        upstream=torch.randn(full.shape,dtype=torch.bfloat16).cuda()*.01
        selection=slice(context.offset,context.offset+65)
        x=full[:,selection].clone().requires_grad_()
        e=upstream[:,selection].contiguous()
        counts=context.valid_counts(None,batch_size=1,device=x.device)
        def step():
            model.zero_grad(set_to_none=True);x.grad=None
            output=model(x,implementation="cuda",sequence_parallel=context,valid_counts=counts)
            output.backward(e);context.reduce_gradients_(model)
            return output
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream): captured=step()
        for _ in range(2):
            with torch.no_grad():
                full.mul_(.9);x.copy_(full[:,selection]);model.core_base_raw.add_(.01)
            graph.replay();torch.cuda.synchronize()
            reference=copy.deepcopy(model);reference.zero_grad(set_to_none=True)
            rx=full.clone().requires_grad_();expected=reference(rx,implementation="cuda")
            expected.backward(upstream)
            assert _relative(captured,expected[:,selection])<.005
            assert _relative(x.grad,rx.grad[:,selection])<.005
            for p,q in zip(model.parameters(),reference.parameters()):assert _relative(p.grad,q.grad)<.005
        graph.reset();graph=None
        del captured,reference,rx,expected
        # Check centered-convolution halos, including gaps at partition borders.
        lengths=(5,8); context=SequenceParallel.create(lengths[rank])
        torch.manual_seed(416)
        conv=_CenteredShortConv(8,3).cuda().double()
        reference=copy.deepcopy(conv)
        full=torch.randn(2,13,8,dtype=torch.float64).cuda()
        upstream=torch.randn_like(full)
        mask=torch.ones(2,13,dtype=torch.bool,device='cuda');mask[0,4]=False;mask[1,5:]=False
        selection=slice(context.offset,context.offset+lengths[rank])
        x=full[:,selection].clone().requires_grad_();rx=full.clone().requires_grad_()
        output=conv(x,mask[:,selection],sequence_parallel=context)
        (output*upstream[:,selection]).sum().backward();context.reduce_gradients_(conv)
        expected=reference(rx,mask);(expected*upstream).sum().backward()
        assert _relative(output,expected[:,selection])<1e-12
        assert _relative(x.grad,rx.grad[:,selection])<1e-12
        for p,q in zip(conv.parameters(),reference.parameters()):assert _relative(p.grad,q.grad)<1e-12
    finally:
        # NCCL graphs retain communicator references, including on assertions.
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<2,reason="requires two CUDA devices")
def test_exact_sequence_shards_match_unsharded_fp64(tmp_path):
    mp.spawn(_worker,args=((tmp_path/'nccl').as_uri(),),nprocs=2,join=True)
