import torch
import torch as th
import torch.nn as nn
from models.softsplat import softsplat
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    zero_module,
)
import torch.nn.functional as F
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock

def compute_mask(flow_bwd_tensor, flow_fwd_tensor):
    metric = torch.ones_like(flow_fwd_tensor[:, :1])
    warped_bwd = softsplat(tenIn=flow_bwd_tensor, tenFlow=flow_fwd_tensor, tenMetric=metric, strMode='soft')
    flow_diff = flow_fwd_tensor + warped_bwd
    occ_mask = (torch.norm(flow_diff, p=2, dim=1, keepdim=True) > 0.3).float()
    return occ_mask

class LocalTimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, local_features=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, LocalResBlock):
                x = layer(x, emb, local_features)
            else:
                x = layer(x)
        return x

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


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


class LocalResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        inject_channels=None
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.norm_in = GroupNorm32(channels)
        self.norm_out = GroupNorm32(self.out_channels)

        self.in_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.Identity(),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, local_conditions):
        return checkpoint(
            self._forward, (x, emb, local_conditions), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, local_conditions):
        h = self.norm_in(x, local_conditions)
        h = self.in_layers(h)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.norm_out(h, local_conditions)
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


def apply_occlusion_mask(warped_img, original_img, occ_mask):
    return warped_img * (1 - occ_mask) + original_img * occ_mask


class FeatureWarperSoftsplat(nn.Module):
    def __init__(self, with_learnable_metric=False, in_channels=128):
        super().__init__()
        self.with_learnable_metric = with_learnable_metric

        if with_learnable_metric:
            # Learn confidence (metric) from input features
            self.metric_net = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
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

        warped = softsplat(
            tenIn=feat_ref,
            tenFlow=flow,
            tenMetric=metric,
            strMode="soft"
        )

        if mask!= None:
             warped = apply_occlusion_mask(warped_img = warped, original_img= feat_ref , occ_mask=mask)
             
        return warped


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


class Occulsion_Aware_FeatureExtractor(nn.Module):

    def __init__(self, inject_channels, dims=2):
        super().__init__()
        self.first_pre_extractor = LocalTimestepEmbedSequential(
            conv_nd(dims, 3, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
        )
        self.last_pre_extractor = LocalTimestepEmbedSequential(
            conv_nd(dims, 3, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
        )
        self.wrapper = FeatureWarperSoftsplat()
        self.extractors_first = nn.ModuleList([
            LocalTimestepEmbedSequential(conv_nd(dims, 32, int(inject_channels[0] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[0] / 2), int(inject_channels[1] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[1] / 2), int(inject_channels[2] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[2] / 2), int(inject_channels[3] / 2), 3, padding=1, stride=2), nn.SiLU())
        ])

        self.extractors_last = nn.ModuleList([
            LocalTimestepEmbedSequential(conv_nd(dims, 32, int(inject_channels[0] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[0] / 2), int(inject_channels[1] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[1] / 2), int(inject_channels[2] / 2), 3, padding=1, stride=2), nn.SiLU()),
            LocalTimestepEmbedSequential(conv_nd(dims, int(inject_channels[2] / 2), int(inject_channels[3] / 2), 3, padding=1, stride=2), nn.SiLU())
        ])
        self.zero_convs = nn.ModuleList([
            zero_module(conv_nd(dims, inject_channels[0], inject_channels[0], 3, padding=1)),
            zero_module(conv_nd(dims, inject_channels[1], inject_channels[1], 3, padding=1)),
            zero_module(conv_nd(dims, inject_channels[2], inject_channels[2], 3, padding=1)),
            zero_module(conv_nd(dims, inject_channels[3], inject_channels[3], 3, padding=1))
        ])

    def forward(self, local_conditions, flow):

        first_frame = local_conditions[:,3:]
        last_frame = local_conditions[:,:3]
        flow_fwd = flow[:,:2]
        flow_bwd = flow[:,2:]

        # print('print shapes:',first_frame.shape, last_frame.shape, flow_fwd.shape,flow_bwd.shape)

        first_features = self.first_pre_extractor(first_frame,None)
        last_features = self.last_pre_extractor(last_frame,None)

        assert len(self.extractors_first) == len(self.zero_convs) == len(self.extractors_last)
        output_features = []

        # normalize and interpolate 
        flow_res = [128, 64, 32 , 16 ]

        for idx in range(len(self.extractors_first)):

            first_features = self.extractors_first[idx](first_features, None)
            last_features = self.extractors_last[idx](last_features, None)

            flow = resize_and_normalize_flow_batched(flow_fwd, target_h=flow_res[idx], target_w=flow_res[idx])
            flow_b = resize_and_normalize_flow_batched(flow_bwd, target_h=flow_res[idx], target_w=flow_res[idx])

            occ_fwd = compute_mask(flow,flow_b)
            occ_bwd = compute_mask(flow_b, flow)

            wrapped_first = self.wrapper(first_features, flow,mask = occ_fwd)
            wrapped_last = self.wrapper(last_features, flow_b,mask = occ_bwd)

            local_features = torch.cat([wrapped_first,wrapped_last],dim=1)

            output_feature = self.zero_convs[idx](local_features)
            output_features.append(output_feature)
        return output_features
