import torch.nn as nn

from . import base

class Conditioner(base.Conditioner):
    def __init__(self,
                 cfg,
                 ):
        super().__init__(cfg)

    def _fix_backbone(self):
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

    def _extract_features(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), 512, 8, 8).contiguous()
        return x

