"""Convert an ESOD zip into the compact event store used for training (no extraction of the inner tar.gz).

Only labeled frames are kept, cut exactly like the released scripts:
  real: each 10 ms npz is split at its mid time (ESOD_real_pt.split_event); the halves are named int((t0+t1)/2)
        and t1, labels come from meta/real_labels.json (converted from all_labels.rar, YOLO-normalized).
  syn : a sequence's npz are sorted by t0 and grouped by 7 (ESOD_syn_pt.save_pt); group count..count+6 is frame
        <count+6>.pt; labels come from pt_index.txt / pt_labels.txt inside the zip.
Raw events (x, y, p, dt) are stored so both TECM variants can be built later (see repro/data.voxelize).

usage:
  python tools/prepare_data.py --dataset real --zip /path/ESOD_real.zip --out /path/prepared/real
  python tools/prepare_data.py --dataset syn  --zip /path/ESOD_syn.zip  --out /path/prepared/syn
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repro.data import EVENT_DTYPE, REAL_CLASSES, SENSOR_H, SENSOR_W, SYN_CLASSES, default_split, seq_key  # noqa: E402

INNER = {"real": "ESOD_real/real_npz.tar", "syn": "ESOD_syn/Syn_data_npz.tar"}


def write_seq(out, seq, frames):
    """frames: list of (frame_t, events (n, 4) int64 [x, y, p, t], boxes (m, 5) [cls, x1, y1, x2, y2])."""
    frames.sort(key=lambda f: f[0])
    offsets = np.zeros(len(frames) + 1, np.int64)
    offsets[1:] = np.cumsum([len(f[1]) for f in frames])
    ev = np.empty(int(offsets[-1]), EVENT_DTYPE)
    span = np.zeros(len(frames), np.int64)
    boxes = []
    for i, (_, e, bx) in enumerate(frames):
        a, b = offsets[i], offsets[i + 1]
        if b > a:
            dt = e[:, 3] - e[0, 3]
            ev["x"][a:b], ev["y"][a:b], ev["p"][a:b], ev["dt"][a:b] = e[:, 0], e[:, 1], e[:, 2], dt
            span[i] = dt[-1]
        for row in bx:
            boxes.append([i, *row])
    boxes = np.asarray(boxes, np.float32).reshape(-1, 6)
    boxes[:, [2, 4]] = boxes[:, [2, 4]].clip(0, SENSOR_W)
    boxes[:, [3, 5]] = boxes[:, [3, 5]].clip(0, SENSOR_H)
    np.save(os.path.join(out, "events", seq + ".npy"), ev)
    np.savez(os.path.join(out, "index", seq + ".npz"), offsets=offsets, span=span, boxes=boxes,
             frame_t=np.asarray([f[0] for f in frames], np.int64))
    return len(frames), int(offsets[-1]), len(boxes)


def iter_tar(zip_path, inner):
    """Yield (sequence folder, file name, events (n, 4) int64 [x, y, p, t]) from the tar.gz inside the zip."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(inner) as raw:
        tf = tarfile.open(fileobj=raw, mode="r|gz")
        for m in tf:
            if m.isfile() and m.name.endswith(".npz"):
                folder, _, name = m.name.rpartition("/")
                with np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=False) as z:
                    yield folder.rsplit("/", 1)[-1], name, z["arr_0"]


def prepare_real(args, stats):
    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)  # {"1": {"11654999": [[cls, xc, yc, w, h], ...]}}

    def px(rows):
        return [[c, (xc - w / 2) * SENSOR_W, (yc - h / 2) * SENSOR_H, (xc + w / 2) * SENSOR_W,
                 (yc + h / 2) * SENSOR_H] for c, xc, yc, w, h in rows]

    cur, frames, done = None, [], set()
    for vid, name, arr in iter_tar(args.zip, INNER["real"]):
        if vid != cur:
            if cur is not None:
                stats[cur] = write_seq(args.out, cur, frames)
                print(f"  {cur}: {stats[cur]}", flush=True)
                done.add(cur)
            if vid in done:
                raise RuntimeError(f"{vid} is not contiguous in the archive")
            cur, frames = vid, []
        lab = labels[vid.replace("video", "")]
        t0, t1 = (int(v) for v in name[:-4].split("_"))
        mid = int((t0 + t1) / 2)  # same rounding as ESOD_real_pt.py
        if str(mid) not in lab and str(t1) not in lab:
            continue
        sp = 0
        if len(arr):
            t = arr[:, 3]
            sp = int(np.argmax(t > (t[0] + t[-1]) / 2))  # first index after the mid time, 0 if none
        if str(mid) in lab:
            frames.append((mid, arr[:sp], px(lab[str(mid)])))
        if str(t1) in lab:
            frames.append((t1, arr[sp:], px(lab[str(t1)])))
    stats[cur] = write_seq(args.out, cur, frames)
    print(f"  {cur}: {stats[cur]}", flush=True)
    n_lab = sum(len(v) for v in labels.values())
    return n_lab


def prepare_syn(args, stats):
    with zipfile.ZipFile(args.zip) as zf:
        idx = [l.strip() for l in zf.read("ESOD_syn/pt_index.txt").decode().splitlines() if l.strip()]
        lab = [l.strip() for l in zf.read("ESOD_syn/pt_labels.txt").decode().splitlines() if l.strip()]
    labels = {}
    for p, l in zip(idx, lab):
        seq, k = p.rsplit("/", 1)
        assert l.split()[0] == k, (p, l)
        labels.setdefault(seq.split("/")[-1], {})[int(k[:-3])] = [
            [float(v) for v in b.split()] for b in re.findall(r"\[([^\]]*)\]", l)]

    def flush(seq, files):
        names = sorted(files, key=lambda x: int(x.split("_")[0]))
        frames = []
        for count in range(0, len(names) - 6, 7):  # ESOD_syn_pt.save_pt: while count < len - 6
            k = count + 6
            if k in labels.get(seq, {}):
                frames.append((k, np.vstack([files[n] for n in names[count:count + 7]]), labels[seq][k]))
        return write_seq(args.out, seq, frames)

    cur, files, done = None, {}, set()
    for seq, name, arr in iter_tar(args.zip, INNER["syn"]):
        if seq != cur:
            if cur is not None:
                stats[cur] = flush(cur, files)
                print(f"  {cur}: {stats[cur]}", flush=True)
                done.add(cur)
            if seq in done:
                raise RuntimeError(f"{seq} is not contiguous in the archive")
            cur, files = seq, {}
        files[name] = arr
    stats[cur] = flush(cur, files)
    print(f"  {cur}: {stats[cur]}", flush=True)
    return len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["real", "syn"], required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=os.path.join(ROOT, "meta", "real_labels.json"), help="real only")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "events"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "index"), exist_ok=True)
    t = time.time()
    stats = {}
    expected = prepare_real(args, stats) if args.dataset == "real" else prepare_syn(args, stats)
    seqs = sorted(stats, key=seq_key)
    n_frames = sum(v[0] for v in stats.values())
    meta = {
        "dataset": args.dataset,
        "classes": REAL_CLASSES if args.dataset == "real" else SYN_CLASSES,
        "source_zip": os.path.basename(args.zip),
        "sequences": {s: {"frames": stats[s][0], "events": stats[s][1], "boxes": stats[s][2]} for s in seqs},
        "split": default_split(args.dataset, seqs),
        "labeled_frames_expected": expected,
        "labeled_frames_found": n_frames,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"done in {time.time() - t:.0f}s: {len(seqs)} sequences, {n_frames} frames "
          f"(expected {expected}), {sum(v[1] for v in stats.values())} events, "
          f"{sum(v[2] for v in stats.values())} boxes")
    if n_frames != expected:
        print("WARNING: labeled frame count differs from the label files")


if __name__ == "__main__":
    main()
