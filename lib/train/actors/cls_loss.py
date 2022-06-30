import torch
import torch.nn as nn
from lib.train.actors.class_matcher import build_matcher


class ClassMixCriterion(nn.Module):
    def __init__(self):
        super(ClassMixCriterion, self).__init__()
        self.matcher = build_matcher()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, logits, gt_bbox, prob_reg_vec, loss_fn):
        logits_tl, logits_br = self._mix_prob_vec(logits, prob_reg_vec)
        loss_tl = self._compute_loss(logits_tl, gt_bbox, loss_fn)
        loss_br = self._compute_loss(logits_br, gt_bbox, loss_fn)
        return loss_tl * 0.5 + loss_br * 0.5

    def _mix_prob_vec(self, logits, prob_reg_vec):
        return logits * prob_reg_vec[0].unsqueeze(-1), logits * prob_reg_vec[1].unsqueeze(-1)

    def _compute_loss(self, src_logits, gt_bbox, loss_fn):
        indices = self.matcher(src_logits, gt_bbox)

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.zeros_like(idx[0], device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], 1,  # 1 : num_classes
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = loss_fn(src_logits.transpose(1, 2), target_classes)
        return loss_ce
