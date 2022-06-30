import torch
from torch import nn

from einops import rearrange, repeat
from lib.train.models.vit_crt.attention import MultiHeadAttention

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ScaledDotProd(nn.Module):
    def __init__(self, dim_head=None, is_attn=False):
        super(ScaledDotProd, self).__init__()
        self.scale = dim_head
        self.is_attn = is_attn
        self.attn = nn.Softmax(-1)

    def forward(self, x1, x2):
        if self.is_attn:
            dots = torch.matmul(x1, x2.transpose(-1, -2)) * self.scale
            return self.attn(dots)
        else:
            return torch.matmul(x1, x2)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_dim, dropout=0., return_last_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.return_last_attn = return_last_attn  # not in use
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(embed_dim, MultiHeadAttention(embed_dim, num_heads=num_heads, dropout=dropout)),
                PreNorm(embed_dim, FeedForward(embed_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask):
        attn_tensor = None

        for d, (attn, ff) in enumerate(self.layers):
            if self.return_last_attn and d == self.depth-1:
                x_new, attn_tensor = attn(x, key_padding_mask=mask, return_attn=self.return_last_attn)
                x = x_new + x
            else:
                x = attn(x, key_padding_mask=mask) + x

            x = ff(x) + x

        return x, attn_tensor


class ViTCR(nn.Module):
    def __init__(self, *, embed_dim, depth, num_heads, mlp_dim, num_patches,
                 dropout=0., emb_dropout=0., compute_mlp_head=False,
                 multiple_mlp_heads=False, mlp_tokens_only=False,
                 return_last_attn=False, mask_token=False):
        super(ViTCR, self).__init__()

        self.compute_mlp_head = compute_mlp_head
        self.multiple_mlp_head = multiple_mlp_heads
        self.mlp_tokens_only = mlp_tokens_only
        self.return_last_attn = return_last_attn

        self.num_tokens = 2 if not mask_token else 3
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, self.embed_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        if mask_token:
            self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        else:
            self.register_parameter("mask_token", None)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embed_dim, depth, num_heads, mlp_dim, dropout, return_last_attn)

        self.to_latent = nn.Identity()
        if self.compute_mlp_head:
            if self.multiple_mlp_head:
                self.cls_mlp_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim)
                )
                self.reg_mlp_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim)
                )
                if self.mask_token is not None:
                    self.mask_mlp_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim)
                )
            else:
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim)
                )

    def forward(self, x, mask=None):
        b, n, _ = x.shape

        mask_token = None

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        reg_tokens = repeat(self.reg_token, '() n d -> b n d', b=b)

        if self.mask_token is not None:
            mask_tokens = repeat(self.mask_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, reg_tokens, mask_tokens,  x), dim=1)
        else:
            x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + self.num_tokens)]
        # Sommo il patch embedding con il positional embedding

        x = self.dropout(x)

        x, attn = self.transformer(x, mask)

        """
        x_cls = x[:, 0]
        x_reg = x[:, 1]
        # print("x pool", x.shape)
        """
        # x = x[:, :2]
        if self.compute_mlp_head:
            if self.multiple_mlp_head:
                cls_token = self.to_latent(x[:, 0])
                reg_token = self.to_latent(x[:, 1])
                cls_token = self.cls_mlp_head(cls_token)
                reg_token = self.reg_mlp_head(reg_token)
                if self.mask_token is not None:
                    mask_token = self.to_latent(x[:, 2])
                    mask_token = self.mask_mlp_head(mask_token)
            elif self.mlp_tokens_only:
                tokens = self.to_latent(x[:, :self.num_tokens])
                tokens = self.mlp_head(tokens)
                cls_token = tokens[:, 0]
                reg_token = tokens[:, 1]
                if self.mask_token is not None:
                    mask_token = tokens[:, 2]
            else:
                x = self.to_latent(x)
                x = self.mlp_head(x)
                cls_token = x[:, 0]
                reg_token = x[:, 1]
        else:
            if self.mask_token is not None:
                cls_token = x[:, 0]
                reg_token = x[:, 1]
                mask_token = x[:, 2]
            else:
                cls_token = x[:, 0]
                reg_token = x[:, 1]

        if self.mask_token is not None:
            feat_mem = x[:, 3:]  # B x HW1+HW2 x EMBED
        else:
            feat_mem = x[:, 2:]

        return cls_token, reg_token, mask_token, feat_mem  # CLS TOKEN - REG TOKEN


def build_transformer(cfg):
    num_patch_search = (cfg.DATA.SEARCH.SIZE / cfg.MODEL.PATCH_SIZE) ** 2
    num_patch_template = (cfg.DATA.TEMPLATE.SIZE / cfg.MODEL.PATCH_SIZE) ** 2
    return ViTCR(
        embed_dim=cfg.MODEL.HIDDEN_DIM,
        mlp_dim=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        depth=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
        num_heads=cfg.MODEL.TRANSFORMER.NHEADS,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        emb_dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        num_patches=int(num_patch_search + num_patch_template),
        compute_mlp_head=cfg.MODEL.TRANSFORMER.MLP_HEAD,
        multiple_mlp_heads=cfg.MODEL.TRANSFORMER.MULTIPLE_MLP_HEADS,
        mlp_tokens_only=cfg.MODEL.TRANSFORMER.MLP_TOKENS_ONLY,
        return_last_attn=cfg.MODEL.TRANSFORMER.RETURN_LAST_ATTN,
        mask_token=cfg.MODEL.HEAD_MASK
    )


if __name__ == '__main__':
    import _init_paths
    from position_encoding import VitPositionalEncoding
    x = torch.rand(7, 3, 20, 20)

    pos_emb = VitPositionalEncoding(patch_size=4, embed_dim=256, in_channels=3)
    emb = pos_emb(x)
    print(emb.shape)

    vit_cr = ViTCR(
        embed_dim=256,
        depth=6,
        num_heads=16,
        mlp_dim=2048
    )

    out = vit_cr(emb)
    print(out[0].shape, out[1].shape)
