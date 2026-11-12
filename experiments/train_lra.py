"""MemSolve in the Nyströmformer LRA backbone and fixed-step training recipe.

Backbone, heads, and OneCycle settings adapted from mlpen/Nystromformer,
commit 56893131bf3fa99b5a2d3ab452a591dca722529a (Apache-2.0).
See THIRD_PARTY_NOTICES.md for source and adaptation details.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from memsolve import MemSolve, MemSolveConfig
from memsolve.ball import cuda
from experiments.sequence_data import DatasetBundle, prepare_lra, collate_tokens, collate_token_pairs
from experiments.train_transformers import (
    _atomic_save, _source_revision, _rng_state, _restore_rng_state, _runtime_metadata,
    _move_batch, _load_toml_defaults, _classification_metrics,
)

UPSTREAM_REVISION = "56893131bf3fa99b5a2d3ab452a591dca722529a"


@dataclass(frozen=True)
class Recipe:
    length: int
    content_length: int
    steps: int
    batch: int
    warmup: int
    eval_every: int


RECIPES = {
    "listops": Recipe(2048, 2000, 5000, 32, 1000, 50),
    "text": Recipe(4096, 4000, 20000, 32, 8000, 500),
    "retrieval": Recipe(4096, 4000, 30000, 32, 800, 300),
    "image": Recipe(1024, 1024, 35000, 256, 175, 175),
    "pathfinder": Recipe(1024, 1024, 62400, 256, 312, 312),
}


class TokenDataset(Dataset):
    """Translate cached EOS-terminated sequences or pixels to the LRA token IDs."""

    def __init__(self, dataset: Dataset, task: str, content_length: int):
        self.dataset, self.task, self.content_length = dataset, task, content_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        if self.task in ("image", "pathfinder"):
            pixels, label = row
            return (pixels.flatten() * 255).round().long(), label
        shift = 2 if self.task in ("text", "retrieval") else 0
        sequences = [tokens[:-1][:self.content_length] - shift for tokens in row[:-1]]
        return (*sequences, row[-1])


def prepare_data(args):
    task = args.task
    bundle = prepare_lra(task, data_root=args.data_root, cache_root=args.cache_root,
        max_length=None if task == "pathfinder" else (1024 if task == "image" else RECIPES[task].length),
        validation_fraction=.1 if task in ("text", "image") else None,
        split_seed=2026 if task in ("text", "image") else None,
        pathfinder_resolution=32 if task == "pathfinder" else None,
        allow_download=False, revision=None, formal=True)
    train, validation, test = bundle.train, bundle.validation, bundle.test
    metadata = dict(bundle.metadata)
    if task == "text":
        # The official LRA/Nyströmformer benchmark uses IMDb test for validation.
        train, validation = bundle.train.dataset, bundle.test
        metadata["split_protocol"] = "official-lra-imdb-test-as-validation"
        metadata.pop("split_fingerprints", None)
        metadata.pop("validation_fraction", None)
        metadata.pop("split_seed", None)
    metadata["input_protocol"] = {
        "padding_id": 0, "eos": False, "mask": "token-id-not-zero",
        "content_length": RECIPES[task].content_length,
        "padded_length": RECIPES[task].length,
        "pixel_encoding": "uint8-token-lookup" if task in ("image", "pathfinder") else None,
    }
    metadata["tokenization"] = ("uint8-pixel-token-lookup" if task in ("image", "pathfinder")
                                else "listops-no-eos" if task == "listops" else "utf8-byte-plus-one-no-eos")
    if task in ("image", "pathfinder"):
        metadata["value_masking"] = "zero-token-masked-as-upstream"
    return DatasetBundle(
        train=TokenDataset(train, task, RECIPES[task].content_length),
        validation=TokenDataset(validation, task, RECIPES[task].content_length),
        test=TokenDataset(test, task, RECIPES[task].content_length),
        input_kind="tokens", num_classes=bundle.num_classes,
        max_length=RECIPES[task].length, vocab_size=32 if task == "listops" else 512,
        pad_token_id=0, paired=bundle.paired, metadata=metadata)


def collate(rows, *, length, paired):
    function = collate_token_pairs if paired else collate_tokens
    batch = function(rows, pad_token_id=0, pad_length=length)
    if paired:
        batch["first_mask"], batch["second_mask"] = batch["first"].ne(0), batch["second"].ne(0)
    else:
        batch["mask"] = batch["inputs"].ne(0)
    return batch


class StepBatchSampler(Sampler):
    """A reproducible stream of shuffled epochs, resumable at any update."""

    def __init__(self, size, batch_size, steps, seed, start_step=0):
        self.size, self.batch_size, self.steps = size, batch_size, steps
        self.seed, self.start_step = seed, start_step

    def __len__(self):
        return self.steps - self.start_step

    def __iter__(self):
        epoch, offset = divmod(self.start_step * self.batch_size, self.size)
        def permutation(e):
            return torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + e)).tolist()
        order = permutation(epoch)
        for _ in range(self.start_step, self.steps):
            indices = []
            while len(indices) < self.batch_size:
                take = min(self.batch_size - len(indices), self.size - offset)
                indices.extend(order[offset:offset + take])
                offset += take
                if offset == self.size:
                    epoch, offset = epoch + 1, 0
                    order = permutation(epoch)
            yield indices


class Block(nn.Module):
    def __init__(self, rank, implementation):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.mixer = MemSolve(MemSolveConfig(dim=64, num_heads=2, rank=rank, bias=True))
        self.implementation = implementation
        self.dropout = nn.Dropout(.1)
        self.mlp = nn.Sequential(nn.Linear(64,128), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(128,64), nn.Dropout(.1))

    def forward(self, x, mask):
        normalized = self.norm1(x)
        if normalized.is_cuda and torch.is_autocast_enabled():
            normalized = normalized.to(torch.get_autocast_dtype("cuda"))
        x = x + self.dropout(self.mixer(normalized, valid_mask=mask, implementation=self.implementation))
        return x + self.mlp(self.norm2(x))


class LRAModel(nn.Module):
    """Nyströmformer's two-layer encoder and single/dual ReLU heads."""

    def __init__(self, length, vocab_size, classes, *, paired=False, rank=32, implementation="cuda"):
        super().__init__()
        self.word_embeddings, self.position_embeddings = nn.Embedding(vocab_size,64), nn.Embedding(length,64)
        nn.init.normal_(self.word_embeddings.weight, std=.02)
        nn.init.normal_(self.position_embeddings.weight, std=.02)
        self.dropout = nn.Dropout(.1)
        self.blocks = nn.ModuleList([Block(rank, implementation) for _ in range(2)])
        self.norm = nn.LayerNorm(64)
        self.paired = paired
        self.classifier = nn.Sequential(nn.Linear(256 if paired else 64,128), nn.ReLU(), nn.Linear(128,classes))

    def encode(self, inputs, mask):
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        x = self.dropout(self.word_embeddings(inputs) + self.position_embeddings(positions)[None])
        for block in self.blocks:
            x = block(x,mask)
        # The upstream mean includes the padded positions after zeroing them.
        return (self.norm(x) * mask[:,:,None]).mean(dim=1)

    def forward(self, inputs, mask, second=None, second_mask=None):
        first = self.encode(inputs,mask)
        if self.paired:
            other = self.encode(second,second_mask)
            first = torch.cat((first,other,first*other,first-other),dim=-1)
        return self.classifier(first)


def forward_batch(model, batch):
    if "first" in batch:
        return model(batch["first"],batch["first_mask"],batch["second"],batch["second_mask"])
    return model(batch["inputs"],batch["mask"])


@torch.no_grad()
def evaluate(model, loader, device, dtype, classes):
    model.eval()
    loss_sum = torch.zeros((),device=device)
    predictions, labels = [], []
    for raw in loader:
        batch = _move_batch(raw,device)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type=="cuda"):
            logits = forward_batch(model,batch)
            loss = F.cross_entropy(logits,batch["labels"],reduction="sum")
        loss_sum += loss
        predictions.append(logits.argmax(-1)); labels.append(batch["labels"])
    y = torch.cat(labels).cpu(); pred = torch.cat(predictions).cpu()
    return {**_classification_metrics(pred,y,classes), "loss":loss_sum.item()/len(y), "examples":len(y)}


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config",type=Path)
    initial,_ = bootstrap.parse_known_args(argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--task",choices=tuple(RECIPES),required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--cache-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--rank",type=int,default=32)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--eval-workers",type=int,default=4)
    parser.add_argument("--execution",choices=("compile","eager"),default="compile")
    parser.add_argument("--amp-dtype",choices=("bfloat16","float16"),default="bfloat16")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke-steps",type=int,default=0)
    parser.add_argument("--prepare-only",action="store_true")
    if initial.config:
        settings=_load_toml_defaults(initial.config)
        unknown=set(settings)-{a.dest for a in parser._actions}
        if unknown: raise ValueError(f"unknown LRA config settings: {sorted(unknown)}")
        parser.set_defaults(**settings)
    return parser.parse_args(argv)


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device); dtype=getattr(torch,args.amp_dtype)
    source=_source_revision()
    if not args.smoke_steps and (source["git_commit"] is None or source["git_dirty"]):
        raise RuntimeError("full LRA runs require a clean committed source tree")
    bundle=prepare_data(args); recipe=RECIPES[args.task]
    model=LRAModel(recipe.length,bundle.vocab_size,bundle.num_classes,paired=bundle.paired,
                   rank=args.rank,implementation="cuda" if device.type=="cuda" else "reference")
    config={"recipe":asdict(recipe),"upstream_revision":UPSTREAM_REVISION,
            "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "model":{"dim":64,"depth":2,"heads":2,"head_dim":32,"ffn":128,"dropout":.1,
                     "position":"learned-1d-absolute","pooling":"fixed-length-mean",
                     "parameters":sum(p.numel() for p in model.parameters())},
            "dataset":bundle.metadata,"split_sizes":{s:len(getattr(bundle,s)) for s in ("train","validation","test")},
            "source_revision":source,"runtime":_runtime_metadata(device,cuda_enabled=device.type=="cuda"),
            "optimizer":{"name":"fused-AdamW","lr":1e-4,"eps":1e-6,"weight_decay":0.,
                         "scheduler":"OneCycleLR-linear","cycle_momentum":True,
                         "base_momentum":.85,"max_momentum":.95,"div_factor":25.,"final_div_factor":10000.},
            "evaluation":"all validation/test examples, validation accuracy selects best checkpoint",
            "precision":"FP32 parameters/residuals, autocast peripheral layers, native mixed-precision operator"}
    if args.prepare_only:
        print(json.dumps(config,indent=2),flush=True);return
    output=args.output; output.mkdir(parents=True,exist_ok=True)
    last=output/"last.pt"; state=None
    identity={k:v for k,v in config.items() if k not in ("runtime",)}
    identity["arguments"]={k:v for k,v in identity["arguments"].items() if k!="resume"}
    if last.exists():
        if not args.resume:raise FileExistsError(f"{last} exists; use --resume or a new output")
        state=torch.load(last,map_location="cpu",weights_only=False)
        if state["identity"]!=identity:raise ValueError("resume configuration differs from saved run")
        if state.get("completed"):print(json.dumps(state["test"]),flush=True);return
    (output/"config.json").write_text(json.dumps(config,indent=2))
    if device.type=="cuda":cuda.load(device=device)
    model.to(device)
    if args.execution=="compile":
        import torch._functorch.config as fc
        fc.backward_pass_autocast="off"
        model.compile(dynamic=True,fullgraph=False,
            options={"triton.cudagraphs":False,"emulate_precision_casts":True})
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-6,
                               weight_decay=0.,fused=device.type=="cuda")
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-4,
        pct_start=recipe.warmup/recipe.steps,anneal_strategy="linear",total_steps=recipe.steps)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and dtype==torch.float16)
    start=0;best_accuracy=-1.;best_loss=float("inf")
    if state:
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
        start=state["step"];best_accuracy,best_loss=state["best_accuracy"],state["best_loss"]
        _restore_rng_state(state["rng"])
    steps=min(recipe.steps,args.smoke_steps) if args.smoke_steps else recipe.steps
    sampler=StepBatchSampler(len(bundle.train),recipe.batch,steps,args.seed,start)
    collate_fn=functools.partial(collate,length=recipe.length,paired=bundle.paired)
    common={"collate_fn":collate_fn,"pin_memory":device.type=="cuda"}
    train_loader=DataLoader(bundle.train,batch_sampler=sampler,num_workers=args.workers,
                            persistent_workers=args.workers>0,**common)
    def evaluation_loader(dataset):
        return DataLoader(dataset,batch_size=recipe.batch,num_workers=args.eval_workers,
                          persistent_workers=args.eval_workers>0,drop_last=False,**common)
    validation_loader=evaluation_loader(bundle.validation);test_loader=evaluation_loader(bundle.test)
    model.train();optimizer.zero_grad(set_to_none=True)
    started=time.perf_counter();loss_sum=torch.zeros((),device=device);correct=torch.zeros((),device=device)
    interval=0;interval_started=started
    for step,raw in enumerate(train_loader,start=start+1):
        batch=_move_batch(raw,device)
        with torch.autocast(device.type,dtype=dtype,enabled=device.type=="cuda"):
            logits=forward_batch(model,batch);loss=F.cross_entropy(logits,batch["labels"])
        scaler.scale(loss).backward()
        scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<scale:raise FloatingPointError("AMP overflow skipped a fixed-budget optimizer update")
        scheduler.step();optimizer.zero_grad(set_to_none=True)
        loss_sum+=loss.detach();correct+=(logits.detach().argmax(-1)==batch["labels"]).float().mean();interval+=1
        if step%50==0 or step==start+1:
            status={"state":"training","step":step,"steps":steps,
                    "elapsed_seconds":time.perf_counter()-started,"updated":time.time(),
                    "train_loss":loss.item(),"lr":optimizer.param_groups[0]["lr"]}
            (output/"status.json").write_text(json.dumps(status))
            print(json.dumps(status),flush=True)
        if step%recipe.eval_every==0 or step==steps:
            training_seconds=time.perf_counter()-interval_started
            validation=evaluate(model,validation_loader,device,dtype,bundle.num_classes)
            improved=(validation["accuracy"],-validation["loss"])>(best_accuracy,-best_loss)
            if improved:best_accuracy,best_loss=validation["accuracy"],validation["loss"]
            metrics={"step":step,"train_loss":loss_sum.item()/interval,"train_accuracy":correct.item()/interval,
                     "validation":validation,"best_accuracy":best_accuracy,"checkpoint_selected":improved,
                     "training_seconds":training_seconds,"training_steps":interval,
                     "elapsed_seconds":time.perf_counter()-started,
                     "lr":optimizer.param_groups[0]["lr"],"beta1":optimizer.param_groups[0]["betas"][0],
                     "peak_gb":torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else 0.}
            if not math.isfinite(metrics["train_loss"]) or not math.isfinite(validation["loss"]):
                raise FloatingPointError("non-finite training or validation loss")
            with (output/"metrics.jsonl").open("a") as f:f.write(json.dumps(metrics)+"\n")
            print(json.dumps(metrics),flush=True)
            state={"identity":identity,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                   "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"step":step,
                   "best_accuracy":best_accuracy,"best_loss":best_loss,"rng":_rng_state()}
            if improved:_atomic_save(state,output/"best.pt")
            _atomic_save(state,last)
            loss_sum.zero_();correct.zero_();interval=0;model.train();interval_started=time.perf_counter()
    best=torch.load(output/"best.pt",map_location="cpu",weights_only=False);model.load_state_dict(best["model"])
    result={**evaluate(model,test_loader,device,dtype,bundle.num_classes),"selected_step":best["step"]}
    (output/"test_metrics.json").write_text(json.dumps(result,indent=2))
    state["completed"]=True;state["test"]=result;_atomic_save(state,last)
    (output/"status.json").write_text(json.dumps({"state":"completed","result":result,"updated":time.time()}))
    print(json.dumps({"test":result}),flush=True)


if __name__=="__main__":
    run(parse_args())
