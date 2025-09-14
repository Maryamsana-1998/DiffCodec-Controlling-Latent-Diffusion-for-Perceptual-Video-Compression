import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

from diffusers.models.controlnets.controlnet import ControlNetModel
from controlnet.extractors import Bi_Dir_ResidueExtractor , WarpExtractor
from controlnet.control_utils import FDN


class ResControlNet(ControlNetModel):
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

        self.feature_extractor = Bi_Dir_ResidueExtractor(inject_channels =self.inject_channels, )
        self.warp_extractor = WarpExtractor(inject_channels =self.inject_channels, )
        self.controlnet_cond_embedding = nn.Identity()

        C64,C32,C16,C08 = self.inject_channels
        self.fdn64 = FDN(norm_nc=C64,  label_nc=C64)   # e.g., C64=320
        self.fdn32 = FDN(norm_nc=C32,  label_nc=C32)   # e.g., C32=640
        self.fdn16 = FDN(norm_nc=C16,  label_nc=C16)   # e.g., C16=1280
        self.fdn08 = FDN(norm_nc=C08,  label_nc=C08)   # e.g., C08=1280


    # ---------- main forward ----------
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: Optional[torch.FloatTensor] = None,        # [B,6,H,W] optional
        flow_cond: Optional[torch.FloatTensor] = None,              # [B,4,H,W] optional
        warp_cond: Optional[torch.FloatTensor] = None,              # [B,3,H,W] optional
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
        prev_frame, next_frame = controlnet_cond[:, :3], controlnet_cond[:, 3:]
        print(prev_frame.shape, next_frame.shape)
        fwd , bwd = flow_cond[:, :2], flow_cond[:, 2:]
        P64, P32, P16, P08 = self.feature_extractor(prev_frame, next_frame, fwd, bwd)
        W64, W32, W16, W08 = self.warp_extractor(warp_cond)
        assert P64.shape[-1] == 64 and P32.shape[-1] == 32 and P16.shape[-1] == 16 and P08.shape[-1] == 8


        # ---- mirrored ControlNet path with injections ----
        sample = self.conv_in(sample)            # [B, 320, 64, 64]
        sample = self.fdn64(sample, P64 + W64) 
        
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
                sample = self.fdn32(sample, P32+ W32)
            elif i == 1:
                # print('add 16')
                sample = self.fdn16(sample, P16+ W16)
            else:
                # print('add 08')
                sample = self.fdn08(sample, P08+ W08)
           
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

        if self.config.global_pool_conditions:
            down_block_res_samples = [x.mean(dim=(2, 3), keepdim=True) for x in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample.mean(dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        # Keep API parity if you prefer returning a dataclass; otherwise this tuple is fine.
        return down_block_res_samples, mid_block_res_sample
