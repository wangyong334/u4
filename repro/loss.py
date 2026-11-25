"""YOLOv8/11 detection loss: task-aligned assigner + BCE (cls) + CIoU (box) + DFL.

Ported from ultralytics 8.3 (utils/loss.py v8DetectionLoss, utils/tal.py TaskAlignedAssigner) so the repo does
not depend on ultralytics. Defaults are ultralytics': topk 10, alpha 0.5, beta 6.0, gains box 7.5 / cls 0.5 /
dfl 1.5, reg_max 16. The paper only says it uses the YOLOv11 head and loss; the gains are not given there.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDES = (8, 16, 32)


def feat_shapes(img_size=640):
    return [(img_size // s, img_size // s) for s in STRIDES]


def make_anchors(shapes, strides, device, dtype=torch.float32, offset=0.5):
    """Anchor centres in grid units, row-major per level (same order as FDDNet's flattened tokens)."""
    points, stride_t = [], []
    for (h, w), s in zip(shapes, strides):
        sy, sx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype) + offset,
                                torch.arange(w, device=device, dtype=dtype) + offset, indexing="ij")
        points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_t.append(torch.full((h * w, 1), s, device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_t)


def dist2bbox(distance, anchor_points):
    lt, rb = distance.chunk(2, -1)
    return torch.cat((anchor_points - lt, anchor_points + rb), -1)


def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


def bbox_ciou(box1, box2, eps=1e-7):
    """CIoU of xyxy boxes (broadcasting), shape (..., 1)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk, self.num_classes, self.alpha, self.beta, self.eps = topk, num_classes, alpha, beta, eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs, self.n_max_boxes = pd_scores.shape[0], gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            return (torch.full_like(pd_scores[..., 0], self.num_classes), torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores), torch.zeros_like(pd_scores[..., 0]).bool(),
                    torch.zeros_like(pd_scores[..., 0]))
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                                                      mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        align_metric *= mask_pos
        pos_align = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align = (align_metric * pos_overlaps / (pos_align + self.eps)).amax(-2).unsqueeze(-1)
        return target_labels, target_bboxes, target_scores * norm_align, fg_mask.bool(), target_gt_idx

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        ind0 = torch.arange(self.bs, device=gt_labels.device).view(-1, 1).expand(-1, self.n_max_boxes)
        ind1 = gt_labels.squeeze(-1).long()
        bbox_scores[mask_gt] = pd_scores[ind0, :, ind1][mask_gt]
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = bbox_ciou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)
        return bbox_scores.pow(self.alpha) * overlaps.pow(self.beta), overlaps

    def select_topk_candidates(self, metrics, topk_mask):
        _, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        topk_idxs.masked_fill_(~topk_mask, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for k in range(self.topk):
            count.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        batch_ind = torch.arange(self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx].clamp_(0)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        target_scores = torch.zeros((*target_labels.shape, self.num_classes), dtype=torch.int64,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        return target_labels, target_bboxes, torch.where(fg > 0, target_scores, 0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # an anchor assigned to several boxes keeps the one with the highest IoU
            multi = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            is_max = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max.scatter_(1, overlaps.argmax(1).unsqueeze(1), 1)
            mask_pos = torch.where(multi, is_max, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        return mask_pos.argmax(-2), fg_mask, mask_pos


class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        return loss_iou, loss_dfl.sum() / target_scores_sum


class DetectionLoss:
    """Loss for FDDNet outputs (B, 8400, 4 * reg_max + nc), tokens ordered P3, P4, P5 row-major."""

    def __init__(self, nc, device, img_size=640, reg_max=16, box=7.5, cls=0.5, dfl=1.5, topk=10):
        self.nc, self.reg_max, self.device, self.img_size = nc, reg_max, device, img_size
        self.gains = (box, cls, dfl)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.assigner = TaskAlignedAssigner(topk=topk, num_classes=nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max).to(device)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)
        self.anchor_points, self.stride_tensor = make_anchors(feat_shapes(img_size), STRIDES, device)

    def preprocess(self, targets, batch_size):
        """targets (n, 6) [b, cls, xc, yc, w, h] normalized -> (B, max_n, 5) [cls, x1, y1, x2, y2] in pixels."""
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)
        i = targets[:, 0].long()
        out = torch.zeros(batch_size, int(torch.bincount(i, minlength=batch_size).max()), 5, device=self.device)
        for j in range(batch_size):
            m = i == j
            n = int(m.sum())
            if n:
                out[j, :n] = targets[m, 1:]
        xywh = out[..., 1:5] * self.img_size
        out[..., 1:5] = torch.cat((xywh[..., :2] - xywh[..., 2:] / 2, xywh[..., :2] + xywh[..., 2:] / 2), -1)
        return out

    def bbox_decode(self, pred_dist):
        b, a, c = pred_dist.shape
        dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(dist, self.anchor_points)

    def __call__(self, preds, targets):
        """Returns (loss * batch_size, detached [box, cls, dfl]) exactly like ultralytics."""
        bs = preds.shape[0]
        pred_distri, pred_scores = preds.float().split((4 * self.reg_max, self.nc), -1)
        gt = self.preprocess(targets.to(self.device), bs)
        gt_labels, gt_bboxes = gt.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(pred_distri)  # grid units
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * self.stride_tensor).type(gt_bboxes.dtype),
            self.anchor_points * self.stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        loss = torch.zeros(3, device=self.device)
        loss[1] = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, self.anchor_points,
                                              target_bboxes / self.stride_tensor, target_scores,
                                              target_scores_sum, fg_mask)
        loss = loss * torch.tensor(self.gains, device=self.device)
        return loss.sum() * bs, loss.detach()
