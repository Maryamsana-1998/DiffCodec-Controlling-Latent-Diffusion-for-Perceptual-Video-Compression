import torch
import torch.nn as nn
import torch.nn.functional as F
from controlnet.control_utils import zero_module, resize_and_normalize_flow_batched, FeatureWarperSoftsplat , compute_mask 

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.SiLU()
        )
    def forward(self, x):
        return self.block(x)

class WarpExtractor(nn.Module):
    """
    Extracts pyramid features from warped images (fwd/bwd)
    and maps them into inject_channels using zero convs.
    """
    def __init__(self, inject_channels=[320, 320, 640, 1280]):
        super().__init__()
        self.inject_channels = inject_channels  # e.g. [320, 320, 640, 1280]

        # Encoder blocks (downsample progressively)
        self.enc1 = ConvBlock(3, 64, stride=4)    # 512 -> 128
        self.enc2 = ConvBlock(64, 320, stride=2)  # 128 -> 64
        self.enc3 = ConvBlock(320, 320, stride=2) # 64 -> 32
        self.enc4 = ConvBlock(320, 640, stride=2) # 32 -> 16
        self.enc5 = ConvBlock(640, 1280, stride=2)# 16 -> 8

        # Zero convs to align feature dims with inject_channels
        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(320,  inject_channels[0], 3, padding=1)),
            zero_module(nn.Conv2d(320,  inject_channels[1], 3, padding=1)),
            zero_module(nn.Conv2d(640,  inject_channels[2], 3, padding=1)),
            zero_module(nn.Conv2d(1280, inject_channels[3], 3, padding=1))
        ])

    def forward(self, x):
        f1 = self.enc1(x)        
        f2 = self.enc2(f1)       # 64×64, 320ch
        f3 = self.enc3(f2)       # 32×32, 320ch
        f4 = self.enc4(f3)       # 16×16, 640ch
        f5 = self.enc5(f4)       # 8×8, 1280ch

        # Apply zero conv projection
        out_feats = [
            self.zero_convs[0](f2),
            self.zero_convs[1](f3),
            self.zero_convs[2](f4),
            self.zero_convs[3](f5)
        ]

        return out_feats

class Bi_Dir_ResidueExtractor(nn.Module):
    """
    Produces a pyramid of residual-conditioning features for diffusion conditioning.
    Inputs (forward pass):
      - prev_frame, next_frame: [B,3,512,512]
      - flow_fwd, flow_bwd: [B,2,512,512] (pixel displacements on full-res)
      - masks: optional occlusion masks at full-res (or None)
    Outputs:
      - list of 4 tensors [B, C64, 64,64], [B, C32,32,32], [B, C16,16,16], [B, C8,8,8]
        where C* = inject_channels
    """
    def __init__(self, inject_channels):
        super().__init__()
        # inject_channels e.g. [320,320,640,1280]
        self.inject = inject_channels
        self.split_res = [int(i//2) for i in inject_channels]  # the internal feature widths before zero-conv
        c1,c2,c3,c4 = self.split_res

        # Separate pre-extractors for prev & next (keeps symmetry but independent params)
        # They produce intermediate small-channel feature that will be further reduced to c1..c4
        self.prev_pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.SiLU(),  # 512 -> 256
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.SiLU(),
        )
        self.next_pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.SiLU(),
        )

        # Pyramid feature extractors (progressive downsample)
        # Note: we produce exactly the internal channel counts c1..c4 at scales 64,32,16,8
        self.prev_pyramids = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, c1, 3, padding=1, stride=2), nn.SiLU()),  # -> 128 -> 64
            nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1, stride=2), nn.SiLU()),  # -> 32
            nn.Sequential(nn.Conv2d(c2, c3, 3, padding=1, stride=2), nn.SiLU()),  # -> 16
            nn.Sequential(nn.Conv2d(c3, c4, 3, padding=1, stride=2), nn.SiLU()),  # -> 8
        ])
        self.next_pyramids = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, c1, 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(c2, c3, 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(c3, c4, 3, padding=1, stride=2), nn.SiLU()),
        ])

        # Learnable downsamplers for flows: we first bilinear interpolate the raw flow
        # to target resolution then pass through a small conv to refine (learned).
        self.flow_refiners = nn.ModuleList([
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2),  # refine for 64x64
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2),  # refine for 32x32
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2),  # refine for 16x16
            nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2),  # refine for 8x8
        ])

        # Optional additional flow feature encoders (for gating/attention) - small MLP convs
        self.flow_feature_encoders = nn.ModuleList([
            nn.Conv2d(2, 16, 3, padding=1),
            nn.Conv2d(2, 16, 3, padding=1),
            nn.Conv2d(2, 32, 3, padding=1),
            nn.Conv2d(2, 32, 3, padding=1),
        ])

        # Warpers that accept features and flow maps (in pixel units for that resolution)
        self.warpers = nn.ModuleList([
            FeatureWarperSoftsplat(with_learnable_metric=True, in_channels=c1),
            FeatureWarperSoftsplat(with_learnable_metric=True, in_channels=c2),
            FeatureWarperSoftsplat(with_learnable_metric=True, in_channels=c3),
            FeatureWarperSoftsplat(with_learnable_metric=True, in_channels=c4),
        ])

        # Zero-initialized convs to expand internal channels to inject_channels
        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(c1, inject_channels[0], kernel_size=3, padding=1)),
            zero_module(nn.Conv2d(c2, inject_channels[1], kernel_size=3, padding=1)),
            zero_module(nn.Conv2d(c3, inject_channels[2], kernel_size=3, padding=1)),
            zero_module(nn.Conv2d(c4, inject_channels[3], kernel_size=3, padding=1)),
        ])

        # list of target resolutions for convenience
        self.resolutions = [64, 32, 16, 8]

    def forward(self, prev_frame, next_frame, flow_fwd, flow_bwd, masks=None):
        """
        prev_frame, next_frame: [B,3,512,512]
        flow_fwd, flow_bwd: [B,2,512,512] (pixel displacements on full-res)
        masks: optional list of occlusion masks at full-res or None
        returns: list of 4 residual-conditioning features aligned with inject_channels
        """
        B, _, H, W = prev_frame.shape
        assert H == 512 and W == 512, "expects 512x512 inputs"

        # 1) Pre-extractors
        prev_pre = self.prev_pre(prev_frame)   # [B,64,256,256]
        next_pre = self.next_pre(next_frame)   # [B,64,256,256]

        # 2) Build pyramid features for prev & next (internal channel widths)
        prev_feats = []
        next_feats = []
        x_prev = prev_pre
        x_next = next_pre
        for enc_prev, enc_next in zip(self.prev_pyramids, self.next_pyramids):
            x_prev = enc_prev(x_prev)   # downsample progressively
            x_next = enc_next(x_next)
            prev_feats.append(x_prev)
            next_feats.append(x_next)
        # prev_feats/next_feats are [64,32,16,8] feature maps with channels c1..c4
        print("Extracted pyramid features:", [f.shape for f in prev_feats])
        # 3) For each target resolution: create flow map (learnable refiner), warp features, fuse, project
        out_feats = []
        for i, res in enumerate(self.resolutions):
            # interpolate to the required spatial size (learnable refinement after)
            # note: flows are pixel displacements in full-res coords; when downsampling spatially,
            # the magnitude should be scaled to the smaller grid. Easiest: interpolate then divide by factor.
            factor = H // res  # e.g., 512/64 = 8
            flow_f_ds = F.interpolate(flow_fwd, size=(res, res), mode='bilinear', align_corners=False) / factor
            flow_b_ds = F.interpolate(flow_bwd, size=(res, res), mode='bilinear', align_corners=False) / factor

            # apply small learnable conv to refine the downsampled flow
            flow_f_ds = self.flow_refiners[i](flow_f_ds)
            flow_b_ds = self.flow_refiners[i](flow_b_ds)

            occ_f = compute_mask(flow_f_ds, flow_b_ds)
            occ_b = compute_mask(flow_b_ds, flow_f_ds)

            # warp the prev/next features (features were computed at the same res)
            feat_prev = prev_feats[i]
            feat_next = next_feats[i]

            warped_prev, conf_prev = self.warpers[i](feat_prev, flow_f_ds, mask=occ_f)
            warped_next, conf_next = self.warpers[i](feat_next, flow_b_ds, mask=occ_b)
            # Soft fusion using confidences (learned by warper) — same as soft_fuse
            conf = torch.cat([conf_prev, conf_next], dim=1)   # [B,2,res,res]
            conf = torch.clamp(conf, min=0.0)
            w_sum = conf.sum(dim=1, keepdim=True) + 1e-6
            w_norm = conf / w_sum
            fused = w_norm[:, :1] * warped_prev + w_norm[:, 1:] * warped_next
            out = self.zero_convs[i](fused)   # [B, inject_C, res, res]
            out_feats.append(out)

        return out_feats

class Bi_Dir_FeatureExtractor(nn.Module):

    def __init__(self, inject_channels):
        super().__init__()
        self.inject = inject_channels
        self.split_res = [int(i/2) for i in self.inject]
        self.first_pre_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.last_pre_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.wrapper = nn.ModuleList([FeatureWarperSoftsplat(with_learnable_metric=True,in_channels=self.split_res[0]), 
                        FeatureWarperSoftsplat(with_learnable_metric=True,in_channels=self.split_res[1]),
                        FeatureWarperSoftsplat(with_learnable_metric=True,in_channels=self.split_res[2]),
                        FeatureWarperSoftsplat(with_learnable_metric=True,in_channels=self.split_res[3])])
        
        self.extractors_first = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, int(inject_channels[0] / 2), 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(int(inject_channels[0] / 2), int(inject_channels[1] / 2), 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(int(inject_channels[1] / 2), int(inject_channels[2] / 2), 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(int(inject_channels[2] / 2), int(inject_channels[3] / 2), 3, padding=1, stride=2), nn.SiLU())
        ])
        self.extractors_last = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, int(inject_channels[0] / 2), 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(int(inject_channels[0] / 2), int(inject_channels[1] / 2), 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(int(inject_channels[1] / 2), int(inject_channels[2] / 2), 3, padding=1, stride=2), nn.SiLU()),
            nn.Sequential(nn.Conv2d(int(inject_channels[2] / 2), int(inject_channels[3] / 2), 3, padding=1, stride=2), nn.SiLU())
        ])

        self.zero_convs = nn.ModuleList([
            zero_module(nn.Conv2d(inject_channels[0]//2, inject_channels[0], 3, padding=1)),
            zero_module(nn.Conv2d(inject_channels[1]//2, inject_channels[1], 3, padding=1)),
            zero_module(nn.Conv2d(inject_channels[2]//2, inject_channels[2], 3, padding=1)),
            zero_module(nn.Conv2d(inject_channels[3]//2, inject_channels[3], 3, padding=1))
        ])
    
    def forward(self, local_conditions, flow):

        first_frame = local_conditions[:,3:]
        last_frame = local_conditions[:,:3]
        flow_fwd = flow[:,:2]
        flow_bwd = flow[:,2:]

        first_features = self.first_pre_extractor(first_frame)
        last_features = self.last_pre_extractor(last_frame)
        
        assert len(self.extractors_first) == len(self.zero_convs) == len(self.extractors_last)
        output_features = []

        # normalize and interpolate 
        flow_res = [64,32,16, 8]

        for idx in range(len(self.extractors_first)):
            # Extract features
            first_features = self.extractors_first[idx](first_features)
            last_features  = self.extractors_last[idx](last_features)

            # Resize flows
            flow_f = resize_and_normalize_flow_batched(flow_fwd, flow_res[idx], flow_res[idx])
            flow_b = resize_and_normalize_flow_batched(flow_bwd, flow_res[idx], flow_res[idx])

            # Occlusion masks
            occ_fwd = compute_mask(flow_f, flow_b)
            occ_bwd = compute_mask(flow_b, flow_f)

            # Warp both sides
            warped_first, conf_fwd = self.wrapper[idx](first_features, flow_f, mask=occ_fwd)
            warped_last,  conf_bwd = self.wrapper[idx](last_features,  flow_b, mask=occ_bwd)

            # === Fusion step (soft_fuse logic) ===
            eps = 1e-6
            conf = torch.cat([conf_fwd, conf_bwd], dim=1)   # [B,2,H,W]
            conf = torch.clamp(conf, min=0)
            w_sum = conf.sum(dim=1, keepdim=True) + eps
            w_norm = conf / w_sum

            fused = w_norm[:, :1] * warped_first + w_norm[:, 1:] * warped_last

            # Handle double holes (both invalid)
            holes = (occ_fwd + occ_bwd) > 1.5   # both occluded
            if holes.any():
                avg = 0.5 * (warped_first + warped_last)
                fused = torch.where(holes.expand_as(fused), avg, fused)

            # Final conv after fusion
            output_feature = self.zero_convs[idx](fused)
            output_features.append(output_feature)

        return output_features