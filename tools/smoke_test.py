"""Smoke test for the released ESOD / FDDNet model: environment, forward/backward,
known gradient issues, training-step memory and streaming inference speed.

Run from the repo root (the folder that contains model/):
    python tools/smoke_test.py                       # server: paper config 2x8 fp32 + bf16 variants
    python tools/smoke_test.py --configs 1x8:bf16    # 8 GB laptop GPU (Windows spills instead of OOM)
The last line is "SMOKE OK" when the model builds, runs forward/backward on the GPU and
returns the expected shapes. Size / grad / memory / speed lines are informational.
"""
import argparse
import os
import platform
import sys
import time

sys.dont_write_bytecode = True  # keep the official model/ folder free of new .pyc files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))  # official code uses flat imports

import torch  # noqa: E402

NUM_CLASS = 11
REF_PARAMS_M = 3.708  # released code, num_class=11 (paper Tab.1 reports 4.0M)
OUT_SHAPE = (1, 8400, 64 + NUM_CLASS)  # 80^2 + 40^2 + 20^2 tokens, 4x16 DFL bins + classes
STATE_SHAPES = [(1, 56, 80, 80), (1, 112, 40, 40), (1, 224, 20, 20)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--configs", default="2x8:fp32,2x8:bf16,1x8:bf16",
                   help="training-step configs 'BxL:fp32|bf16', comma separated; empty string skips")
    p.add_argument("--iters", type=int, default=3, help="timed training iterations per config (after 1 warm-up)")
    p.add_argument("--fps-iters", type=int, default=50)
    p.add_argument("--no-flops", action="store_true", help="skip the thop count")
    return p.parse_args()


def grad_sum(module):
    grads = [p.grad for p in module.parameters()]
    if all(g is None for g in grads):
        return None
    return sum(float(g.abs().sum()) for g in grads if g is not None)


def train_step(FDDNet, dev, B, L, amp, iters):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    model = FDDNet(num_class=NUM_CLASS).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xs = torch.randn(L, B, 10, 640, 640, device=dev)
    times = []
    try:
        for i in range(1 + iters):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            state, loss = None, 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                for k in range(L):  # BPTT over the clip, state carried between segments
                    preds, state = model(xs[k], state)
                    loss = loss + preds.float().pow(2).mean()
            loss.backward()
            opt.step()
            torch.cuda.synchronize(dev)
            if i > 0:
                times.append(time.perf_counter() - t0)
        it = sum(times) / len(times)
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
        resv = torch.cuda.max_memory_reserved(dev) / 2**30
        return f"{it:.2f} s/iter ({it / (B * L) * 1000:.0f} ms/frame), peak alloc {peak:.2f} GB, reserved {resv:.2f} GB"
    except RuntimeError as ex:
        if "out of memory" in str(ex).lower():
            return "OOM"
        raise
    finally:
        del model, opt, xs
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    fails = []
    print(f"python {platform.python_version()} | torch {torch.__version__} | CUDA build {torch.version.cuda}")
    major, minor = (int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    if (major, minor) < (1, 10):
        print("SMOKE FAIL: torch >= 1.10 required (ESOD_model.py calls torch.meshgrid(..., indexing='ij'))")
        return 1
    if not torch.cuda.is_available():
        print("SMOKE FAIL: CUDA is not available")
        return 1
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    prop = torch.cuda.get_device_properties(dev)
    free, total = torch.cuda.mem_get_info(dev)
    print(f"GPU {prop.name} (sm_{prop.major}{prop.minor}) | {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free")
    import einops
    import torchvision
    print(f"torchvision {torchvision.__version__} | einops {einops.__version__}")
    from ESOD_model import FDDNet  # also imports thop, Defor_transformer, Mamba

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)

    # 1. model size vs paper
    n = sum(p.numel() for p in FDDNet(num_class=NUM_CLASS).parameters())
    tag = "same as released code" if abs(n / 1e6 - REF_PARAMS_M) < 0.005 else "DIFFERS from released code"
    print(f"[size] params {n / 1e6:.3f} M ({tag}; paper 4.0 M)")
    if not args.no_flops:
        from thop import profile
        m = FDDNet(num_class=NUM_CLASS).eval()
        with torch.no_grad():
            macs, _ = profile(m, inputs=(torch.randn(1, 10, 640, 640),), verbose=False)
        print(f"[size] thop MACs {macs / 1e9:.2f} G (paper 'FLOPs' 5.8 G)")
        del m

    # 2. forward / backward on the GPU, state carried over two segments
    model = FDDNet(num_class=NUM_CLASS).to(dev).train()
    x = torch.randn(1, 10, 640, 640, device=dev)
    p1, s1 = model(x)
    p2, _ = model(x, s1)
    got = (tuple(p1.shape), [tuple(s.shape) for s in s1])
    ok = got == (OUT_SHAPE, STATE_SHAPES)
    print(f"[fwd] output {got[0]}, state {got[1]} -> {'OK' if ok else 'UNEXPECTED'}")
    if not ok:
        fails.append("output/state shapes")
    (p1.float().pow(2).mean() + p2.float().pow(2).mean()).backward()
    finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    print(f"[bwd] gradients finite: {finite}")
    if not finite:
        fails.append("non-finite gradients")

    # 3. known issues of the released code (informational): None / 0 means "never trained"
    head = model.transformer.encoder1
    ldtr = model.LDTR.mamba_layers[0].defor
    for name, mod in [
        ("enc_score_head (top-300 query selection)", model.enc_score_head),
        ("head deform-attn sampling_offsets", head.self_attn.sampling_offsets),
        ("head deform-attn ref_point_proj", head.ref_point_proj),
        ("LDTR(s3) align sampling_offsets", ldtr.self_attn.sampling_offsets),
        ("LDTR(s3) align ref_point_proj", ldtr.ref_point_proj),
        ("head deform-attn attention_weights (control)", head.self_attn.attention_weights),
    ]:
        g = grad_sum(mod)
        print(f"[grad] {name:46s} {'None' if g is None else f'{g:.3e}'}")
    del model, x, p1, p2, s1
    torch.cuda.empty_cache()

    # 4. training-step memory / speed (dummy loss; the real YOLO loss adds a little)
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        bl, prec = cfg.split(":")
        B, L = (int(v) for v in bl.lower().split("x"))
        res = train_step(FDDNet, dev, B, L, prec == "bf16", args.iters)
        torch.cuda.empty_cache()
        print(f"[train] B={B} L={L} {prec}: {res}")

    # 5. streaming inference speed (B=1, fp32, state carried, no NMS)
    model = FDDNet(num_class=NUM_CLASS).to(dev).eval()
    x = torch.randn(1, 10, 640, 640, device=dev)
    with torch.no_grad():
        state = None
        for _ in range(10):
            _, state = model(x, state)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(args.fps_iters):
            _, state = model(x, state)
        torch.cuda.synchronize(dev)
    print(f"[infer] {args.fps_iters / (time.perf_counter() - t0):.1f} FPS (B=1, fp32, no NMS; paper 88 FPS on V100)")

    print("SMOKE OK" if not fails else f"SMOKE FAIL: {', '.join(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
