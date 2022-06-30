# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from torch import nn
import numpy as np


class TrackingMatcher(nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, pred_logits, gt_bbox):
        """ Performs the matching

        Params:
            pred_logits: This is a Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

            gt_bbox: This is a list of targets (len(targets) = batch_size), where each target
                bounding boxes Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        # print("pred_logits", pred_logits.shape)
        bs, num_queries = pred_logits.shape[:2]
        for i in range(bs):
            # print("gt_box", gt_bbox[i].shape)
            cx, cy, w, h = [x.item() for x in gt_bbox[i]]
            xmin, ymin, xmax, ymax = cx-w/2, cy-h/2, cx+w/2, cy+h/2

            len_feature = int(np.sqrt(num_queries))
            Xmin = int(np.ceil(xmin*len_feature))
            Ymin = int(np.ceil(ymin*len_feature))
            Xmax = int(np.ceil(xmax*len_feature))
            Ymax = int(np.ceil(ymax*len_feature))

            if Xmin == Xmax:
                Xmax = Xmax + 1
            if Ymin == Ymax:
                Ymax = Ymax + 1
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            c = b[Ymin:Ymax, Xmin:Xmax].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c, d)
            indices.append(indice)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher():
    return TrackingMatcher()
