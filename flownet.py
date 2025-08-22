import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

from diffusers.models.controlnets.controlnet import ControlNetModel

# replace this with your own import
from softsplat import softsplat  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

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

        warped = softsplat(
            tenIn=feat_ref,
            tenFlow=flow,
            tenMetric=metric,
            strMode="soft"
        )

        if mask!= None:
             warped = warped*(1-mask)
             
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

class Bi_Dir_FeatureExtractor(nn.Module):

    def __init__(self, inject_channels):
        super().__init__()
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
        self.wrapper = FeatureWarperSoftsplat()
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
            zero_module(nn.Conv2d(inject_channels[0], inject_channels[0], 3, padding=1)),
            zero_module(nn.Conv2d(inject_channels[1], inject_channels[1], 3, padding=1)),
            zero_module(nn.Conv2d(inject_channels[2], inject_channels[2], 3, padding=1)),
            zero_module(nn.Conv2d(inject_channels[3], inject_channels[3], 3, padding=1))
        ])
    
    def forward(self, local_conditions, flow):

        first_frame = local_conditions[:,3:]
        last_frame = local_conditions[:,:3]
        flow_fwd = flow[:,:2]
        flow_bwd = flow[:,2:]

        # print('print shapes:',first_frame.shape, last_frame.shape, flow_fwd.shape,flow_bwd.shape)

        first_features = self.first_pre_extractor(first_frame)
        last_features = self.last_pre_extractor(last_frame)
        
        assert len(self.extractors_first) == len(self.zero_convs) == len(self.extractors_last)
        output_features = []

        # normalize and interpolate 
        flow_res = [64,32,16, 8]

        for idx in range(len(self.extractors_first)):

            first_features = self.extractors_first[idx](first_features)
            last_features = self.extractors_last[idx](last_features)

            flow = resize_and_normalize_flow_batched(flow_fwd, target_h=flow_res[idx], target_w=flow_res[idx])
            flow_b = resize_and_normalize_flow_batched(flow_bwd, target_h=flow_res[idx], target_w=flow_res[idx])

            wrapped_first = self.wrapper(first_features, flow)
            wrapped_last = self.wrapper(last_features,flow_b)

            local_features = torch.cat([wrapped_first,wrapped_last],dim=1)
            # print('local shape', local_features.shape)

            out_features = self.zero_convs[idx](local_features)
            output_features.append(out_features)
        return output_features

class DualFlowControlNet(ControlNetModel):
    """
    A "simple" ControlNet that:
      - takes TWO control images (I0, I1) and TWO flows (I0->I1, I1->I0),
      - builds a pyramid of control features,
      - warps features per scale with scaled flows,
      - injects them into the ControlNet down path (additive bias),
      - returns standard ControlNet residuals for the main UNet.

    Compatible with diffusers UNet via (down_block_additional_residuals, mid_block_additional_residual).
    """

    def __init__(
        self,
        *args,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        **kwargs,
    ):
        """
        Args:
            conditioning_channels: channels for each control image (3 for RGB).
            fuse_width: internal feature width for the control pyramid at the base scale.
            block_out_channels: must match the paired UNet's config (same as ControlNetModel).
        """
        super().__init__(*args, block_out_channels=block_out_channels, **kwargs)

        self.block_out_channels = block_out_channels  # keep for clarity
        self.inject_channels = (320, 320, 640 , 1280)

        self.feature_extractor = Bi_Dir_FeatureExtractor(inject_channels =self.inject_channels, )
        self.controlnet_cond_embedding = nn.Identity()

        C64,C32,C16,C08 = self.inject_channels
        self.fdn64 = FDN(norm_nc=C64,  label_nc=C64)   # e.g., C64=320
        self.fdn32 = FDN(norm_nc=C32,  label_nc=C32)   # e.g., C32=640
        self.fdn16 = FDN(norm_nc=C16,  label_nc=C16)   # e.g., C16=1280
        self.fdn08 = FDN(norm_nc=C08,  label_nc=C08)   # e.g., C08=1280

        # self.zero_convs = nn.ModuleList([
        #     zero_module(nn.Conv2d(C64, C64, 1)),
        #     zero_module(nn.Conv2d(C32, C32, 1)),
        #     zero_module(nn.Conv2d(C16, C16, 1)),
        #     zero_module(nn.Conv2d(C08, C08, 1)),
        # ])




    # ---------- main forward ----------
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.FloatTensor] = None,        # [B,6,H,W] optional
        flow_cond: Optional[torch.FloatTensor] = None,               # [B,4,H,W] optional
        conditioning_scale: float = 1.0,
        guess_mode: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        """
        Returns (down_block_res_samples, mid_block_res_sample) just like ControlNetModel.
        """
        # ---- timestep embedding ----
        if not torch.is_tensor(timestep):
            dtype = torch.float32 if sample.device.type == "mps" else torch.float64 if isinstance(timestep, float) else torch.int64
            timestep = torch.tensor([timestep], dtype=dtype, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timesteps = timestep.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)

        # ---- pyramid control (warped & projected) ----
        P64, P32, P16, P08 = self.feature_extractor(controlnet_cond, flow_cond)
        assert P64.shape[-1] == 64 and P32.shape[-1] == 32 and P16.shape[-1] == 16 and P08.shape[-1] == 8


        # ---- mirrored ControlNet path with injections ----
        sample = self.conv_in(sample)            # [B, 320, 64, 64]

        # print('sample ', torch.std(sample), torch.mean(sample))

        # print(torch.std(P64), torch.mean(P64))
        sample = self.fdn64(sample, P64) 

        # print('sample after fdn64 ', torch.std(sample), torch.mean(sample))
        # sample = sample + self.zero_convs[0](P64)  # additive residual  
        sample = sample + P64
        
        down_block_res_samples: Tuple[torch.Tensor, ...] = (sample,)
        for i, down_block in enumerate(self.down_blocks,):
            # standard ControlNet down pass with time/text
            if getattr(down_block, "has_cross_attention", False):
                sample, res_samples = down_block(
                    hidden_states=sample, temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    **{k: v for k, v in kwargs.items() if k in ("attention_mask", "cross_attention_kwargs")}
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)

            if i == 0:
                # print('add 32')
                sample = self.fdn32(sample, P32)
                # sample = sample + self.zero_convs[1](P32)
                sample = sample + P32
            elif i == 1:
                # print('add 16')
                sample = self.fdn16(sample, P16)
                # sample = sample + self.zero_convs[2](P16)
                sample = sample + P16
            else:
                # print('add 08')
                sample = self.fdn08(sample, P08)
                # sample = sample + self.zero_convs[3](P08)
                sample = sample + P08

                
            down_block_res_samples += res_samples


        # mid block
        if getattr(self.mid_block, "has_cross_attention", False):
            sample = self.mid_block(
                sample, emb, encoder_hidden_states=encoder_hidden_states,
                **{k: v for k, v in kwargs.items() if k in ("attention_mask", "cross_attention_kwargs")}
            )
        else:
            sample = self.mid_block(sample, emb)

        controlnet_down_block_res_samples: Tuple[torch.Tensor, ...] = ()
        for down_res, ctrl_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            controlnet_down_block_res_samples += (ctrl_block(down_res),)

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # --- scaling (like ref) ---
        down_block_res_samples = [x * conditioning_scale for x in controlnet_down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        # # ---- scale like ControlNet ----
        # if guess_mode and not self.config.global_pool_conditions:
        #     # Optional logspace per-stage scaling (0.1..1.0), same as diffusers
        #     import math
        #     n = len(controlnet_down_block_res_samples) + 1
        #     scales = torch.logspace(math.log10(0.1), 0, n, device=sample.device) * conditioning_scale
        #     down_block_res_samples = [x * s for x, s in zip(controlnet_down_block_res_samples, scales[:-1])]
        #     mid_block_res_sample = mid_block_res_sample * scales[-1]
        # else:
        #     down_block_res_samples = [x * conditioning_scale for x in controlnet_down_block_res_samples]
        #     mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [x.mean(dim=(2, 3), keepdim=True) for x in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample.mean(dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        # Keep API parity if you prefer returning a dataclass; otherwise this tuple is fine.
        return down_block_res_samples, mid_block_res_sample
