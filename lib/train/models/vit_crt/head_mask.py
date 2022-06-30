import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskHead(nn.Module):
    def __init__(self, input_size=320, in_channels=256):
        super(MaskHead, self).__init__()
        self.input_size = input_size

        self.v0 = nn.Sequential(
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v1 = nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v2 = nn.Sequential(
                nn.Conv2d(512, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h2 = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h1 = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h0 = nn.Sequential(
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, corr_feat, Lfeat):
        """
            corr_feat: (batch,256,20,20)
            Lfeat: (batch,64,160,160), (batch,256,80,80), (batch,512,40,40)
        """
        h2_size = int(self.input_size / 8)
        h1_size = int(self.input_size / 4)
        h0_size = int(self.input_size / 2)

        out = self.post0(F.interpolate(self.h2(corr_feat), size=(h2_size, h2_size),
                                       mode='bilinear', align_corners=False)
                         + self.v2(Lfeat[2]))  # (b,32,32,32)+(b,32,32,32) --> (b,16,32,32)

        out = self.post1(F.interpolate(self.h1(out), size=(h1_size, h1_size),
                                       mode='bilinear', align_corners=False)
                         + self.v1(Lfeat[1]))  # (b,16,64,64)+(b,16,64,64) --> (b,4,64,64)

        out = self.post2((F.interpolate(self.h0(out), size=(h0_size, h0_size),
                                        mode='bilinear', align_corners=False)
                          + self.v0(Lfeat[0])).contiguous())
        out = torch.sigmoid(F.interpolate(out, size=(self.input_size, self.input_size),
                                          mode='bilinear', align_corners=False))  # (b,1,256,256)
        return out


def build_mask_head(cfg):
    if cfg.MODEL.HEAD_MASK:
        return MaskHead(
            input_size=cfg.DATA.SEARCH.SIZE,
            in_channels=cfg.MODEL.HIDDEN_DIM
        )
    else:
        return None
