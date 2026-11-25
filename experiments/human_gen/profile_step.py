"""Where does a human training step spend its time?  (batch 8, full model, original split)
1) data loading alone: 60 batches from the train loader
2) forward+backward on one fixed batch, fp32 and bf16 autocast: time + peak memory
3) torch.profiler top ops by CUDA time, and per-modality timings via forward hooks
"""
import os, sys, time
from experiments.common.paths import split_config
B = os.environ.get("BATCH_SIZE", "8")
sys.argv = ["p", "--config_fname", str(split_config("gene")), "--sfgenes", "493", "--batch_size", B, "--device", "cuda"]
import torch, numpy as np
from args import argparser_fn
from data.splicedata_dataloader import splicedata_dataloader
from utils.validation import model_perd
from model import model_registry
args = argparser_fn(dataset_type="gtex", batch_size=int(B)); dev = torch.device("cuda")
data = splicedata_dataloader(args); data.setup(); args = data.setup_hparams(args)
loader = data.train_dataloader()
t = time.time(); it = iter(loader); batches = [next(it) for _ in range(60)]
print(f"[prof] data: {60 / (time.time() - t):.2f} batches/s (60 batches, batch {B})", flush=True)
model = model_registry.str2model("deltacellsplicenet")(args).to(dev)
print(f"[prof] params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
opt = torch.optim.Adam(model.parameters(), lr=2e-5)
meta, x, y = batches[0]
print("[prof] input shapes:", [tuple(v.shape) for v in x], flush=True)
def step(amp):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        _, loss, _ = model_perd(model, x, meta, y, dev, embed_list_return=False)
    opt.zero_grad(); loss.backward(); opt.step()
for amp in (False, True):
    try:
        for _ in range(3): step(amp)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time()
        for _ in range(10): step(amp)
        torch.cuda.synchronize()
        print(f"[prof] amp={amp}: {(time.time() - t) / 10 * 1000:.0f} ms/step, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"[prof] amp={amp}: OOM", flush=True); torch.cuda.empty_cache()
# per-module forward timing
enc = model.psi_transformer_encoder
times = {}
def hook_pre(name):
    def f(m, i): torch.cuda.synchronize(); times.setdefault(name, []).append(-time.time())
    return f
def hook_post(name):
    def f(m, i, o): torch.cuda.synchronize(); times[name][-1] += time.time()
    return f
for name, mod in enc.named_children():
    mod.register_forward_pre_hook(hook_pre(name)); mod.register_forward_hook(hook_post(name))
for _ in range(5):
    with torch.no_grad(): model_perd(model, x, meta, y, dev, embed_list_return=False)
print("[prof] forward time by encoder child (ms):", {k: round(float(np.mean(v[1:])) * 1000, 1) for k, v in times.items()}, flush=True)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step(False)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20), flush=True)
