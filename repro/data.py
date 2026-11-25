"""Event store, training clips and GPU-side TECM for the ESOD datasets.

Prepared layout (written by tools/prepare_data.py):
  <root>/meta.json          dataset, classes, sequences, split
  <root>/events/<seq>.npy   packed events of all labeled frames: x, y (uint16), p (uint8, 0/1 as recorded),
                            dt (uint32, microseconds from the first event of the frame)
  <root>/index/<seq>.npz    offsets (n+1), frame_t (n), span (n), boxes (m, 6) [frame, cls, x1, y1, x2, y2] in px
"""
import json
import os
import random

import numpy as np
import torch

EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("dt", "<u4")])  # packed, 9 bytes/event
SENSOR_W, SENSOR_H = 1280, 720
IN_SIZE = 640  # paper: event frames are resized (stretched) to 640 x 640
NUM_BINS = 5   # TECM = 2 polarities x 5 bins = 10 input channels

REAL_CLASSES = ["human", "airplane", "bird", "insect", "car", "pingpong"]
SYN_CLASSES = ["airplane", "ball", "bike", "boat", "bird", "car", "human", "insect", "motorcycle", "rocket", "UAV"]
REAL_TO_SYN = [6, 0, 4, 7, 5, 1]  # real class id -> syn class id (pingpong -> ball), used for syn2real


def seq_key(s):
    return [int(v) for v in s.replace("video", "").split("_")]


def default_split(dataset, seqs):
    """Fixed 7:3 split by recording unit (our choice; the paper does not publish its split)."""
    if dataset == "real":  # 20 videos, each contains all 6 classes
        test = {f"video{i}" for i in (3, 6, 9, 12, 15, 18)}
    else:  # 80 base videos x 4 variants that share the same objects -> split by base video
        test = {s for s in seqs if int(s.split("_")[0]) % 10 in (2, 5, 8)}
    return {"train": sorted((s for s in seqs if s not in test), key=seq_key),
            "test": sorted((s for s in seqs if s in test), key=seq_key)}


class EventStore:
    """Read-only access to a prepared dataset; event files are memory-mapped and opened lazily."""

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self._ev, self._idx = {}, {}

    def __getstate__(self):  # do not pickle open memmaps into DataLoader workers
        d = self.__dict__.copy()
        d["_ev"], d["_idx"] = {}, {}
        return d

    def index(self, seq):
        if seq not in self._idx:
            with np.load(os.path.join(self.root, "index", seq + ".npz")) as z:
                self._idx[seq] = {k: z[k] for k in z.files}
        return self._idx[seq]

    def num_frames(self, seq):
        return len(self.index(seq)["frame_t"])

    def frames(self, seq, start, stop):
        """Events of frames [start, stop): int32 (n, 4) [x, y, p, dt] and frame id (n,) relative to start."""
        if seq not in self._ev:
            self._ev[seq] = np.load(os.path.join(self.root, "events", seq + ".npy"), mmap_mode="r")
        off = self.index(seq)["offsets"]
        a = np.asarray(self._ev[seq][off[start]:off[stop]])
        ev = np.empty((len(a), 4), np.int32)
        ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3] = a["x"], a["y"], a["p"], a["dt"]
        fid = np.repeat(np.arange(stop - start, dtype=np.int32), np.diff(off[start:stop + 1]))
        return ev, fid

    def boxes(self, seq, start, stop):
        """(m, 6) float32 [frame - start, cls, x1, y1, x2, y2] in sensor pixels."""
        b = self.index(seq)["boxes"]
        lo, hi = np.searchsorted(b[:, 0], [start, stop])  # boxes are sorted by frame
        out = b[lo:hi].copy()
        out[:, 0] -= start
        return out


def make_clips(store, seqs, L, rng):
    """Non-overlapping runs of L consecutive labeled frames, random phase per sequence (one pass per epoch)."""
    clips = []
    for s in seqs:
        n = store.num_frames(s)
        if n >= L:
            o = rng.randrange(L)
            clips += [(s, st) for st in range(o, n - L + 1, L)]
    rng.shuffle(clips)
    return clips


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, store, clips, L):
        self.store, self.clips, self.L = store, clips, L

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        seq, st = self.clips[i]
        ev, fid = self.store.frames(seq, st, st + self.L)
        return ev, fid, self.store.boxes(seq, st, st + self.L)


def collate(batch, L):
    """-> events (N, 4) int32, frame id (N,) int32 in [0, B*L), boxes (M, 7) [b, l, cls, x1, y1, x2, y2]."""
    evs, fids, boxes = [], [], []
    for b, (ev, fid, bx) in enumerate(batch):
        evs.append(ev)
        fids.append(fid + b * L)
        out = np.empty((len(bx), 7), np.float32)
        out[:, 0], out[:, 1:] = b, bx
        boxes.append(out)
    return (torch.from_numpy(np.concatenate(evs)), torch.from_numpy(np.concatenate(fids)),
            torch.from_numpy(np.concatenate(boxes)))


def voxelize(ev, fid, n_frames, mode, dataset):
    """TECM on the GPU: ev (N, 4) long [x, y, p, dt], fid (N,) long -> (n_frames, 10, 640, 640) float counts.

    mode "fixed"   : as the paper describes - both polarities, 5 equal time bins over the frame's event span.
    mode "official": as the released ESOD_*_pt.py compute - bin = floor(4 * (t - first) / max(70000, last - first)),
                     first/last taken per polarity; ESOD-real keeps only p == 1 (the script tests p < 0 on 0/1
                     polarities). The release's int8 cast is not reproduced (per-pixel counts stay far below 127).
    Both modes sum events into the 640 x 640 grid (x * 640 // 1280, y * 640 // 720) instead of resizing a dense map.
    """
    dev = ev.device
    x = ev[:, 0] * IN_SIZE // SENSOR_W
    y = ev[:, 1] * IN_SIZE // SENSOR_H
    p, dt = ev[:, 2], ev[:, 3]
    keep = None
    if mode == "fixed":
        span = torch.zeros(n_frames, dtype=torch.long, device=dev).scatter_reduce_(0, fid, dt, "amax")
        b = (dt * NUM_BINS // span[fid].clamp(min=1)).clamp_(max=NUM_BINS - 1)
    elif mode == "official":
        g = fid * 2 + (1 - p)
        first = torch.full((n_frames * 2,), 2**62, dtype=torch.long, device=dev).scatter_reduce_(0, g, dt, "amin")
        last = torch.zeros(n_frames * 2, dtype=torch.long, device=dev).scatter_reduce_(0, g, dt, "amax")
        b = (NUM_BINS - 1) * (dt - first[g]) // (last[g] - first[g]).clamp(min=70000)
        if dataset == "real":
            keep = p == 1
    else:
        raise ValueError(f"unknown TECM mode {mode}")
    idx = ((fid * (2 * NUM_BINS) + (1 - p) * NUM_BINS + b) * IN_SIZE + y) * IN_SIZE + x
    if keep is not None:
        idx = idx[keep]
    vox = torch.zeros(n_frames * 2 * NUM_BINS * IN_SIZE * IN_SIZE, device=dev)
    vox.index_add_(0, idx, torch.ones(idx.numel(), device=dev))
    return vox.view(n_frames, 2 * NUM_BINS, IN_SIZE, IN_SIZE)


def frame_targets(boxes, b_l):
    """boxes (M, 7) [b, l, cls, x1, y1, x2, y2] px -> (m, 6) [b, cls, xc, yc, w, h] normalized, for frame b_l."""
    t = boxes[boxes[:, 1] == b_l]
    xc = (t[:, 3] + t[:, 5]) / (2 * SENSOR_W)
    yc = (t[:, 4] + t[:, 6]) / (2 * SENSOR_H)
    w = (t[:, 5] - t[:, 3]) / SENSOR_W
    h = (t[:, 6] - t[:, 4]) / SENSOR_H
    return torch.stack((t[:, 0], t[:, 2], xc, yc, w, h), 1)


def seeded_rng(seed, epoch):
    return random.Random(seed * 100003 + epoch)
