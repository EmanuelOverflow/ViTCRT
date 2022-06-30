"""
Basic ViTCRT Model (Spatial-only).
"""
import torch
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .vit_transformer import build_transformer
from .head import build_box_head, build_cls_head
from lib.utils.box_ops import box_xyxy_to_cxcywh

from einops.layers.torch import Rearrange


class ViTCRT(nn.Module):
    """ This is the base class for ViTCRT """
    def __init__(self, backbone, transformer, box_head, cls_head, patch_size,
                 head_type="CORNER", token_attn=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.cls_head = cls_head
        self.token_attn = token_attn
        hidden_dim = transformer.embed_dim
        self.mask_rearrange = Rearrange('b (h p1) (w p2) -> b (p1 p2 h w)', p1=patch_size, p2=patch_size)
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, img=None, seq_dict=None, mode="backbone", search_image=False, cls_scores=True):
        if mode == "backbone":
            return self.forward_backbone(img, search_image=search_image)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, cls_scores=cls_scores)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor, search_image=False):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_transformer(self, seq_dict, cls_scores=True):
        # Forward the transformer encoder and decoder
        cls_token, reg_token, mask_token, feat_repr = self.transformer(seq_dict["feat_pos"], seq_dict["mask"])
        # Forward the corner head
        cls_head_exists = self.cls_head is not None
        # Box Head ###
        out, outputs_coord = self.forward_box_head(reg_token, feat_repr, return_dist=cls_head_exists)
        # CLS Head ###
        if cls_head_exists and cls_scores:
            cls_out = self.forward_cls_head(cls_token, feat_repr)
            out.update(cls_out)
        # MASK Head ###
        return out, outputs_coord

    def head_attention_token(self, token, compact_repr):
        """
            hs: output embeddings (B, C)
            memory: encoder embeddings (B, HW1+HW2, C)
        """
        enc_opt = compact_repr[:, -self.feat_len_s:, :]  # encoder output for the search region (B, HW, C)
        token_opt = token.unsqueeze(-1)  # (B, C, N)
        att = torch.matmul(enc_opt, token_opt)  # (B, HW, N)
        # BxCxN * BxNxHW = BxCxHW
        opt = (token_opt * att.transpose(-1, -2)).unsqueeze(1).contiguous()  # (B, C, HW) --> (B, N, C, HW)
        return opt

    def forward_box_head(self, reg_token, compact_repr, return_dist=False):
        if self.token_attn:
            opt = self.head_attention_token(reg_token, compact_repr)
        else:
            opt = self.head_attention(reg_token, compact_repr)

        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # run the corner head
        coords, prob_vec_tl, prob_vec_br = self.box_head(opt_feat, return_dist=return_dist)
        outputs_coord = box_xyxy_to_cxcywh(coords)
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new, 'prob_vec_tl': prob_vec_tl, 'prob_vec_br': prob_vec_br}
        return out, outputs_coord_new

    def forward_cls_head(self, cls_token, compact_repr):
        opt = self.head_attention_token(cls_token, compact_repr)
        opt = opt.squeeze(1).transpose(-1, -2)  # B x HW x C
        logits = self.cls_head(opt)
        out = {'pred_logits': logits}
        return out

    def adjust(self, output_back: list, pos_embed: list):
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        pos_embed_vec = pos_embed[-1]  # BxHWxC
        mask_vec = None
        return {"feat_pos": pos_embed_vec, "mask": mask_vec}


def build_vitcrt(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    box_head = build_box_head(cfg)
    cls_head = build_cls_head(cfg)
    model = ViTCRT(
        backbone,
        transformer,
        box_head,
        patch_size=cfg.MODEL.PATCH_SIZE,
        head_type=cfg.MODEL.HEAD_TYPE,
        cls_head=cls_head,
        token_attn=cfg.MODEL.TOKEN_ATTN
    )

    return model
