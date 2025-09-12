import torch
import torch.nn as nn
import torch.nn.functional as F
from controlnet.softsplat import softsplat  

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def compute_mask(flow_bwd_tensor, flow_fwd_tensor):
    metric = torch.ones_like(flow_fwd_tensor[:, :1])
    with torch.cuda.amp.autocast(enabled=False):
        warped_bwd = softsplat(tenIn=flow_bwd_tensor, tenFlow=flow_fwd_tensor, tenMetric=metric, strMode='soft')
    flow_diff = flow_fwd_tensor + warped_bwd
    occ_mask = (torch.norm(flow_diff, p=2, dim=1, keepdim=True) > 0.3).float()
    return occ_mask

class FDN(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        pw = ks // 2
        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)
        self.conv_gamma = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)
        self.conv_beta = nn.Conv2d(label_nc, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, local_features):
        normalized = self.param_free_norm(x)
        assert local_features.size()[2:] == x.size()[2:]
        gamma = self.conv_gamma(local_features)
        beta = self.conv_beta(local_features)
        out = normalized * (1 + gamma) + beta
        return out

class FeatureWarperSoftsplat(nn.Module):
    def __init__(self, with_learnable_metric=False, in_channels=128):
        super().__init__()
        self.with_learnable_metric = with_learnable_metric

        if with_learnable_metric:
            # Learn confidence (metric) from input features
            self.metric_net = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output: [B, 1, H, W]
            )

    def forward(self, feat_ref, flow, mask= None):
        """
        feat_ref: tensor of shape [B, C=128, H=128, W=128]
        flow:     tensor of shape [B, 2, H, W] (optical flow in pixel units)
        """
        if self.with_learnable_metric:
            metric = self.metric_net(feat_ref)  # [B, 1, H, W]
        else:
            # Default: uniform confidence
            metric = torch.ones_like(flow[:, :1])  # shape: [B, 1, H, W]

        # print('metric', metric.shape)
        with torch.cuda.amp.autocast(enabled=False):
            warped = softsplat(
                tenIn=feat_ref,
                tenFlow=flow,
                tenMetric=metric,
                strMode="soft"
            )

        if mask!= None:
             warped = warped*(1-mask)
             
        return warped, metric

def resize_and_normalize_flow_batched(flow_tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize and normalize batched 2D optical flow tensors for warping.

    Args:
        flow_tensor (torch.Tensor): Flow tensor of shape [B, 2, H, W].
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        torch.Tensor: Normalized flow tensor of shape [B, 2, target_h, target_w].
    """
    # Resize with bilinear interpolation
    resized = F.interpolate(flow_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)

    # Prepare normalization grid
    norm_w = (target_w - 1) / 2.0
    norm_h = (target_h - 1) / 2.0

    # Normalize u and v components
    u = resized[:, 0] / norm_w  # [B, target_h, target_w]
    v = resized[:, 1] / norm_h  # [B, target_h, target_w]
    normalized = torch.stack([u, v], dim=1)  # [B, 2, target_h, target_w]

    return normalized