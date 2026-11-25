"""Train / evaluate FDDNet (official, unmodified model code) on a prepared ESOD dataset.

Paper settings (supplementary 1.2): 100 epochs, AdamW lr 1e-3, weight decay 1e-4, 10 warm-up epochs, sequence
length L = 8, batch 2, 640 x 640 input; YOLOv11 loss summed over the L frames of a clip (Eq. 18), state carried
through the clip; inference runs each video front to back.
Our choices where the paper is silent: cosine decay to 1% after warm-up, grad-norm clip 10 (ultralytics),
bf16 autocast, no augmentation, the last-epoch model is reported (no model selection on the test split).

usage:
  python tools/train.py --data <prepared>/real --out run/real_fixed
  python tools/train.py --data <prepared>/real --out run/real_fixed --resume run/real_fixed/last.pt
  python tools/train.py --data <prepared>/real --out run/real_fixed_eval --eval-only run/real_fixed/last.pt
  python tools/train.py --data <prepared>/real --out run/syn2real --eval-only run/syn_fixed/last.pt  # syn2real
"""
import argparse
import json
import math
import os
import sys
import time
from functools import partial

sys.dont_write_bytecode = True  # keep the official model/ folder free of new .pyc files
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "model"))

import torch  # noqa: E402
from ESOD_model import FDDNet  # noqa: E402
from repro.data import (REAL_TO_SYN, ClipDataset, EventStore, collate, frame_targets, make_clips,  # noqa: E402
                        seeded_rng, voxelize)
from repro.evaluate import evaluate  # noqa: E402
from repro.loss import DetectionLoss  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prepared dataset root (tools/prepare_data.py --out)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repr", choices=["fixed", "official"], default="fixed",
                    help="TECM: 'fixed' = as the paper describes, 'official' = as the released *_pt.py compute")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=10)
    ap.add_argument("--amp", choices=["bf16", "none"], default="bf16")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-split", choices=["test", "train"], default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval-only", default="", help="checkpoint to evaluate, no training")
    ap.add_argument("--max-clips", type=int, default=0, help="sanity runs: clips per epoch")
    ap.add_argument("--max-eval-frames", type=int, default=0, help="sanity runs: frames per evaluated sequence")
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def lr_at(it, base, warm, total):
    if it < warm:
        return base * (it + 1) / warm
    p = min(1.0, (it - warm) / max(1, total - warm))
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))


def run_eval(model, store, args, dev, nc, repr_mode, tag):
    seqs = store.meta["split"][args.eval_split]
    class_ids = REAL_TO_SYN if (store.meta["dataset"] == "real" and nc == 11) else None  # syn2real
    t = time.time()
    res = evaluate(model, store, seqs, repr_mode, dev, nc, amp=args.amp == "bf16", class_ids=class_ids,
                   max_frames=args.max_eval_frames or None)
    res.update({"tag": tag, "split": args.eval_split, "data": store.meta["dataset"], "repr": repr_mode,
                "seconds": round(time.time() - t, 1)})
    print("[eval]", json.dumps(res), flush=True)
    with open(os.path.join(args.out, "metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    return res


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    store = EventStore(args.data)
    dataset = store.meta["dataset"]
    print(f"[cfg] {json.dumps(vars(args))}", flush=True)
    print(f"[data] {dataset}: train {len(store.meta['split']['train'])} seqs, test {len(store.meta['split']['test'])}"
          f" seqs, classes {store.meta['classes']}", flush=True)

    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        model = FDDNet(num_class=ck["nc"]).to(dev)
        model.load_state_dict(ck["model"])
        res = run_eval(model, store, args, dev, ck["nc"], ck["args"]["repr"], f"eval_only:{args.eval_only}")
        with open(os.path.join(args.out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        return

    nc = len(store.meta["classes"])
    model = FDDNet(num_class=nc).to(dev)
    crit = DetectionLoss(nc, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    L, B = args.seq_len, args.batch
    train_seqs = store.meta["split"]["train"]
    n_frames = sum(store.num_frames(s) for s in train_seqs)
    per_epoch = (args.max_clips or n_frames // L) // B
    total, warm = per_epoch * args.epochs, int(args.warmup_epochs * per_epoch)
    start_epoch, it = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, it = ck["epoch"], ck["it"]
        print(f"[resume] epoch {start_epoch}, iter {it}", flush=True)
    print(f"[train] {n_frames} train frames, ~{per_epoch} iters/epoch, {total} iters, warm-up {warm}", flush=True)

    amp = args.amp == "bf16"
    for epoch in range(start_epoch, args.epochs):
        clips = make_clips(store, train_seqs, L, seeded_rng(args.seed, epoch))
        if args.max_clips:
            clips = clips[:args.max_clips]
        loader = torch.utils.data.DataLoader(
            ClipDataset(store, clips, L), batch_size=B, shuffle=False, num_workers=args.workers,
            collate_fn=partial(collate, L=L), drop_last=True, pin_memory=True)
        model.train()
        t0, seen, agg = time.time(), 0, torch.zeros(3)
        torch.cuda.reset_peak_memory_stats(dev)
        for ev, fid, boxes in loader:
            lr = lr_at(it, args.lr, warm, total)
            for g in opt.param_groups:
                g["lr"] = lr
            vox = voxelize(ev.to(dev, non_blocking=True).long(), fid.to(dev, non_blocking=True).long(),
                           B * L, args.repr, dataset).view(B, L, 10, 640, 640)
            boxes = boxes.to(dev, non_blocking=True)
            state, loss, items = None, 0.0, torch.zeros(3, device=dev)
            for l in range(L):  # BPTT through the clip, loss summed over frames (paper Eq. 18)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    preds, state = model(vox[:, l], state)
                lo, li = crit(preds, frame_targets(boxes, l))
                loss = loss + lo
                items += li
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            it += 1
            seen += B * L
            agg += (items / L).cpu()
            if it % args.log_every == 0:
                n = args.log_every
                el = time.time() - t0
                print(f"[it {it}] ep {epoch + 1}/{args.epochs} loss box {agg[0] / n:.3f} cls {agg[1] / n:.3f} "
                      f"dfl {agg[2] / n:.3f} | lr {lr:.2e} gnorm {float(gnorm):.2f} | {seen / el:.1f} frames/s "
                      f"| mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB", flush=True)
                agg.zero_()
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch + 1, "it": it, "nc": nc,
              "classes": store.meta["classes"], "args": vars(args)}
        torch.save(ck, os.path.join(args.out, "last.pt.tmp"))
        os.replace(os.path.join(args.out, "last.pt.tmp"), os.path.join(args.out, "last.pt"))
        print(f"[epoch {epoch + 1}] {time.time() - t0:.0f}s, {seen} frames", flush=True)
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            res = run_eval(model, store, args, dev, nc, args.repr, f"epoch{epoch + 1}")
            if epoch + 1 == args.epochs:
                with open(os.path.join(args.out, "final.json"), "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
