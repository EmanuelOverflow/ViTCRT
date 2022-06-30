import os

from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search_vit
from lib.train.actors.cls_loss import ClassMixCriterion
import numpy as np
import cv2
import uuid
from torchvision.utils import save_image, make_grid
from lib.train.actors.class_matcher import build_matcher


class ViTCRTActor(BaseActor):
    """ Actor for training the VIT-CRT"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        if "cls" in objective:
            if self.settings.head_cls_type == "BOX_MIX":
                self.loss_cls = ClassMixCriterion()
            else:
                self.loss_cls_matcher = build_matcher()

    def __call__(self, data, epoch=None):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """

        # forward pass
        out_dict = self.forward_pass(data)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)
        gt_masks = data['search_masks']

        # compute losses
        loss, status = self.compute_losses(out_dict, gt_bboxes[0], gt_masks[0])

        return loss, status

    def forward_pass(self, data):
        feat_dict_list = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            feat_dict_list.append(self.net(img=NestedTensor(template_img_i, template_att_i),
                                           mode='backbone', search_image=False))

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list.append(self.net(img=NestedTensor(search_img, search_att),
                                       mode='backbone', search_image=True))

        # run the transformer and compute losses
        seq_dict = merge_template_search_vit(feat_dict_list)
        out_dict, _ = self.net(seq_dict=seq_dict, mode="transformer")
        return out_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def compute_cls_loss(self, src_logits, gt_bbox, loss_cls):

        indices = self.loss_cls_matcher(src_logits, gt_bbox)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.zeros_like(idx[0], device=src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], 1,  # 1 : num_classes
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = loss_cls(src_logits.transpose(1, 2), target_classes)
        return loss_ce

    def compute_losses(self, pred_dict, gt_bbox, gt_mask=None, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if 'cls' in self.objective:
            if self.settings.head_cls_type == "BOX_MIX":
                loss_cls = self.loss_cls(pred_dict['pred_logits'], gt_bbox,
                                         (pred_dict["prob_vec_tl"], pred_dict["prob_vec_br"]),
                                         self.objective['cls'])
            else:
                loss_cls = self.compute_cls_loss(pred_dict['pred_logits'], gt_bbox, self.objective['cls'])
            loss = loss + loss_cls * self.loss_weight['cls']
        if 'mask' in self.objective:
            mask_pos_1 = gt_mask == 1
            if mask_pos_1.sum() > 0:
                mask_pred = pred_dict['mask'].squeeze(1)
                mask_loss = self.objective['mask'](mask_pred[mask_pos_1], gt_mask[mask_pos_1])
                loss = loss + mask_loss * self.loss_weight['mask']
            else:
                mask_loss = torch.zeros((1,))

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            if "cls" in self.objective:
                status['Loss/cls'] = loss_cls.item()
            if "mask" in self.objective:
                status['Loss/mask'] = mask_loss.item()
            return loss, status
        else:
            return loss
