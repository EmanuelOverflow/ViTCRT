# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 09:44:55 2020
@author: emanueldinardo
"""
import torch
import torch.nn as nn
import warnings


class ScaledDotProd(nn.Module):
    def __init__(self, scale=None, is_attn=False):
        super(ScaledDotProd, self).__init__()
        self.scale = scale
        self.is_attn = is_attn
        self.softmax = nn.Softmax(-1)

    def forward(self, x1, x2):
        if self.is_attn:
            dots = torch.matmul(x1, x2.transpose(-1, -2)) * self.scale
            return self.softmax(dots)
        else:
            return torch.matmul(x1, x2)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            nn.Dropout(dropout)
        )

        self.attn = ScaledDotProd(scale=self.scale, is_attn=True)
        self.out_qk_v = ScaledDotProd()

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, return_attn=False):
        qkv = qkv.transpose(1, 0)
        q = self.query_proj(qkv)
        k = self.key_proj(qkv)
        v = self.value_proj(qkv)

        tgt_len, bsz, embed_dim = qkv.shape
        src_len, _, _ = qkv.shape
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # B, Nt, E = q.shape
        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1,
                                                           src_len)  # NEW .reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # q = q / math.sqrt(E) Scaling forzato sulle membership, valori piccoli
        # print("q", q.shape)
        # print("k", k.shape)
        attn = self.attn(q, k)

        if attn_mask is not None:
            attn += attn_mask

        out = self.out_qk_v(attn, v)
        out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        out = self.out_proj(out).transpose(0, 1)

        if return_attn:
            return out, attn

        return out
