#!/usr/bin/env python3
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from backbones.adm import Artifact_Detection_Module
from backbones.efficientnet_pytorch import EfficientNet


class CADDM(nn.Module):

    def __init__(self, num_classes, backbone='resnet34', svd=False, haad_cfg=None):
        super(CADDM, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.svd = svd

        if backbone == 'efficientnet-b3':
            self.base_model = EfficientNet.from_pretrained(
                'efficientnet-b3', out_size=[1, 3]
            )
        elif backbone == 'efficientnet-b4':
            self.base_model = EfficientNet.from_pretrained(
                'efficientnet-b4', out_size=[1, 3]
            )
        else:
            raise ValueError("Unsupported Backbone!")

        self.inplanes = self.base_model.out_num_features

        self.proj_dim = self.inplanes

        self.adm = Artifact_Detection_Module(self.inplanes)
        if self.svd:
            self.proj = OrthogonalProjection(input_dim=self.inplanes, output_dim=self.proj_dim)

        self.fc = nn.Linear(self.proj_dim, num_classes)

        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def _stack_or_none(self, lst, dim=0):
        # helper to stack possible Nones safely
        if len(lst) == 0: return None
        if lst[0] is None: return None
        return torch.stack(lst, dim=dim)

    def forward(self, img, return_with_haad=False):
        batch_num = img.size(0)

        # 1) backbone
        feat_map, global_feat = self.base_model(img)  # feat_map: [B,C,H,W], global_feat: [B,C,1,1]

        # 2) ADM
        loc, cof, adm_final_feat = self.adm(feat_map)  # adm_final_feat: [B,C,H,W]

        # 3) optional SVD projection on global features
        if self.svd:
            global_feat_new = self.proj(global_feat.view(batch_num, -1))
            global_feat_new = global_feat_new.view(batch_num, -1, 1, 1)
        else:
            global_feat_new = global_feat

        # 4) fuse features for classifier head
        final_cls_feat = global_feat_new + adm_final_feat
        return_final_cls_feat = final_cls_feat

        # 5) classifier
        final_cls = self.fc(final_cls_feat.view(batch_num, -1))

        # ====== TRAIN MODE ======
        if self.training:
            return loc, cof, final_cls, return_final_cls_feat.view(batch_num, -1)

        # ====== EVAL MODE ======
        if not return_with_haad or not self.haad_enabled:
            # 保持原有行为（只返回 logits）
            return final_cls

    def orthogonality_regularization(self):
        return self.proj.orthogonality_loss() if self.svd else torch.tensor(0.0, device=self.fc.weight.device)

# vim: ts=4 sw=4 sts=4 expandtab

class OrthogonalProjection(torch.nn.Module):
    """
    A linear projection layer with orthogonality regularization (SVDNet-style).
    """
    def __init__(self, input_dim, output_dim, rank=None):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(input_dim, output_dim))  # [input_dim, output_dim]
        nn.init.orthogonal_(self.U)
    def forward(self, x):
        h_norm = F.normalize(x, p=2, dim=1)  # [B, input_dim]
        h_proj = torch.matmul(h_norm, self.U)  # [B, proj_dim]
        return h_proj
    def orthogonality_loss(self):
        UU_T = torch.matmul(self.U.t(), self.U)  # [proj_dim, proj_dim]
        I = torch.eye(UU_T.size(0), device=UU_T.device)
        return F.mse_loss(UU_T, I)
