import torch.nn as nn

from . import base

class Conditioner(base.Conditioner):
    def __init__(self,
                 cfg,
                 ):
        super().__init__(cfg)

    def _fix_backbone(self):
        self.backbone.classifier[4] = nn.Identity()
        self.backbone.aux_classifier = nn.Identity()

    def _extract_features(self, x):
        x = self.backbone.backbone(x)
        x = self.backbone.classifier[0](x['out']) # ASPP
        x = self.backbone.classifier[1](x) # Conv
        x = self.backbone.classifier[2](x) # BN
        x = self.backbone.classifier[3](x) # ReLU

        return x

