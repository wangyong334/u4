"""Streaming evaluation: run FDDNet over each test sequence in time order (state carried, as in the paper's
inference), decode YOLO outputs, NMS with ultralytics' val defaults, COCO mAP with pycocotools."""
import contextlib
import io

import numpy as np
import torch
import torchvision

from .data import IN_SIZE, SENSOR_H, SENSOR_W, voxelize
from .loss import STRIDES, dist2bbox, feat_shapes, make_anchors


class Decoder:
    def __init__(self, nc, device, reg_max=16, class_ids=None):
        self.nc, self.reg_max = nc, reg_max
        self.class_ids = None if class_ids is None else torch.as_tensor(class_ids, device=device)
        self.anchors, self.strides = make_anchors(feat_shapes(IN_SIZE), STRIDES, device)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)

    def __call__(self, preds, conf=0.001, iou=0.7, max_det=300, max_nms=30000):
        """preds (1, A, 4*reg_max + nc) -> boxes (k, 4) xyxy in sensor px, scores (k,), labels (k,)."""
        box, cls = preds[0].float().split((4 * self.reg_max, self.nc), -1)
        if self.class_ids is not None:  # e.g. syn2real: keep only the syn classes that exist in ESOD-real
            cls = cls[:, self.class_ids]
        dist = box.view(-1, 4, self.reg_max).softmax(-1).matmul(self.proj)
        xyxy = (dist2bbox(dist, self.anchors) * self.strides).clamp_(0, IN_SIZE)
        scores = cls.sigmoid()
        i, j = (scores > conf).nonzero(as_tuple=True)  # multi-label, as ultralytics val
        s = scores[i, j]
        if s.numel() > max_nms:
            top = s.topk(max_nms).indices
            i, j, s = i[top], j[top], s[top]
        b = xyxy[i]
        keep = torchvision.ops.batched_nms(b, s, j, iou)[:max_det]
        b = b[keep] * torch.tensor([SENSOR_W / IN_SIZE, SENSOR_H / IN_SIZE] * 2, device=b.device)
        return b, s[keep], j[keep]


def coco_map(gts, dets, nc):
    """gts: list per frame of (m, 5) [cls, x1, y1, x2, y2]; dets: (n, 7) [img, x, y, w, h, score, cls].
    Returns mAP@[.5:.95], mAP@.5, mAP@.75 (pycocotools, all areas, maxDets 100)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    anns = []
    for img, g in enumerate(gts):
        for c, x1, y1, x2, y2 in g.tolist():
            anns.append({"id": len(anns) + 1, "image_id": img, "category_id": int(c) + 1, "iscrowd": 0,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)})
    if len(dets) == 0:
        return [0.0, 0.0, 0.0]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": [{"id": i} for i in range(len(gts))], "annotations": anns,
                      "categories": [{"id": c + 1} for c in range(nc)]}
        gt.createIndex()
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return [float(v) for v in ev.stats[:3]]


@torch.no_grad()
def evaluate(model, store, seqs, mode, device, nc, amp=True, class_ids=None, max_frames=None, chunk=32):
    """Run each sequence front to back with the recurrent state; returns metrics and frame count."""
    was_training = model.training
    model.eval()
    dec = Decoder(nc, device, class_ids=class_ids)  # nc = model classes; class_ids selects/reorders them
    dataset = store.meta["dataset"]
    gts, dets, img = [], [], 0
    for seq in seqs:
        n = store.num_frames(seq)
        if max_frames:
            n = min(n, max_frames)
        state = None
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            ev, fid = store.frames(seq, s, e)
            vox = voxelize(torch.from_numpy(ev).to(device).long(), torch.from_numpy(fid).to(device).long(),
                           e - s, mode, dataset)
            boxes = store.boxes(seq, s, e)
            for k in range(e - s):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    preds, state = model(vox[k:k + 1], state)
                b, sc, lb = dec(preds)
                if len(sc):
                    xywh = torch.cat((b[:, :2], b[:, 2:] - b[:, :2]), 1)
                    d = torch.cat((torch.full_like(sc[:, None], img), xywh, sc[:, None], lb[:, None].float() + 1), 1)
                    dets.append(d.cpu().numpy())
                gts.append(boxes[boxes[:, 0] == k][:, 1:])
                img += 1
    model.train(was_training)
    out_nc = nc if class_ids is None else len(class_ids)
    m = coco_map(gts, np.concatenate(dets) if dets else np.zeros((0, 7)), out_nc)
    return {"mAP": m[0], "mAP50": m[1], "mAP75": m[2], "frames": img}
