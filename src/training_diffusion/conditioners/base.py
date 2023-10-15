import torch
import torch.nn as nn
from torchvision import models

class Conditioner(nn.Module):
    def __init__(self,
                 cfg,
                 ):
        super().__init__()
        self.cfg = cfg
        # count number of '.' in cfg.conditioner.backbone and recursively import
        # e.g. cfg.conditioner.backbone = 'resnet.resnet50' will import torchvision.models.resnet.resnet50
        _m = models
        for i in range(cfg.conditioner.backbone.count('.') + 1):
            _m = getattr(_m, cfg.conditioner.backbone.split('.')[i])
        self.backbone = _m(pretrained=True)
        self._fix_backbone()

        self.suffix = nn.Sequential(
            nn.Conv2d(self.cfg.conditioner.backbone_dim, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, cfg.model.dim, 1, 1),
        )

        self.activation = getattr(torch, cfg.conditioner.activation)

        # # -------------------------------------------------------------
        # # Initialize weights
        # # -------------------------------------------------------------
        # for m in self.suffix.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)
        # -------------------------------------------------------------
        
    def _fix_backbone(self):
        raise NotImplementedError

    def _extract_features(self, x):
        raise NotImplementedError

    def forward(self, x):
        x = self._extract_features(x) # (B, C, H, W)
        x = self.suffix(x) # (B, C, H, W)
        x = self.activation(x) # (B, C, H, W)
        x = x.view(x.shape[0], x.shape[1], -1) # (B, C, H*W)
        return x
    

