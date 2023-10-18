import torch.nn as nn
from torchvision.ops import Conv2dNormActivation

from . import base

class Conditioner(base.Conditioner):
    def __init__(self,
                 cfg,
                 ):
        super().__init__(cfg)

    def _fix_backbone(self):
        self.backbone.backbone['0'] = Conv2dNormActivation(
            in_channels=4,
            out_channels=16,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
            bias=False,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.Hardswish
        )
        self.backbone.classifier[4] = nn.Identity()
        self.backbone.aux_classifier = nn.Identity()

    def _extract_features(self, x):
        x = self.backbone.backbone(x)
        x = self.backbone.classifier[0](x['out']) # ASPP
        x = self.backbone.classifier[1](x) # Conv
        x = self.backbone.classifier[2](x) # BN
        x = self.backbone.classifier[3](x) # ReLU

        return x

