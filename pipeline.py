import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from controlnet.flownet import DualFlowControlNet
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.models import AutoencoderKL,  UNet2DConditionModel
from dataclasses import dataclass
from typing import List, Optional, Union
import inspect
import numpy as np
import PIL.Image
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]] 

class StableDiffusionDualFlowControlNetPipeline(
    StableDiffusionControlNetPipeline,
    DiffusionPipeline
):
    """
    Text-to-image pipeline for a ControlNet that expects:
      - controlnet_cond: [B, 6, H, W] (two RGB images concatenated)
      - flow_cond:       [B, 4, H, W] (forward(2) + backward(2) optical flow)
    Works with SD v1.x UNet (in_channels=4). Returns PIL or tensor.
    """

    def __init__( 
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            controlnet:  DualFlowControlNet,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
            ):
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.register_to_config(requires_safety_checker=requires_safety_checker)


    # convenience
    def enable_xformers_memory_efficient_attention(self):
        if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
            self.unet.enable_xformers_memory_efficient_attention()
        if hasattr(self.controlnet, "enable_xformers_memory_efficient_attention"):
            self.controlnet.enable_xformers_memory_efficient_attention()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        controlnet_cond: torch.Tensor = None,       # [B,6,H,W]
        flow_cond: torch.Tensor = None,             # [B,4,H,W]
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        # -------------------------------
        # 0) Basic checks & setup
        # -------------------------------
        device = self._execution_device
        dtype  = self.unet.dtype
        self._cross_attention_kwargs = cross_attention_kwargs
        self._clip_skip = clip_skip
        self._guidance_scale = guidance_scale
        self._interrupt = False

        if controlnet_cond is None or flow_cond is None:
            raise ValueError("Provide both controlnet_cond [B,6,H,W] and flow_cond [B,4,H,W].")
        if controlnet_cond.ndim != 4 or controlnet_cond.shape[1] != 6:
            raise ValueError(f"controlnet_cond must be [B,6,H,W], got {tuple(controlnet_cond.shape)}")
        if flow_cond.ndim != 4 or flow_cond.shape[1] != 4:
            raise ValueError(f"flow_cond must be [B,4,H,W], got {tuple(flow_cond.shape)}")

        # batch size from prompt or embeds
        if prompt_embeds is not None:
            base_batch = prompt_embeds.shape[0]
        elif isinstance(prompt, list):
            base_batch = len(prompt)
        else:
            base_batch = 1

        do_cfg = guidance_scale is not None and guidance_scale > 1.0
        # self.do_classifier_free_guidance = do_cfg  # keep parity with diffusers

        # align control guidance lists
        if not isinstance(control_guidance_start, list):
            control_guidance_start = [control_guidance_start]
        if not isinstance(control_guidance_end, list):
            control_guidance_end = [control_guidance_end]
        # single ControlNet -> length 1
        if len(control_guidance_start) != 1:
            control_guidance_start = [control_guidance_start[0]]
        if len(control_guidance_end) != 1:
            control_guidance_end = [control_guidance_end[0]]

        # -------------------------------
        # 1) Encode prompts (handles CFG)
        # -------------------------------
        text_encoder_lora_scale = (
            self._cross_attention_kwargs.get("scale", None) if self._cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self._clip_skip,
        )
        if do_cfg:
            # concat uncond/cond once, stays [2B, ..., cad]
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # -------------------------------
        # 2) Prepare control tensors
        # -------------------------------
        # broadcast to batch_size * num_images_per_prompt
        B_ctrl, _, Hc, Wc = controlnet_cond.shape
        batch_size = base_batch * num_images_per_prompt
        if B_ctrl != batch_size:
            if B_ctrl == 1:
                controlnet_cond = controlnet_cond.expand(batch_size, -1, -1, -1).contiguous()
                flow_cond       = flow_cond.expand(batch_size, -1, -1, -1).contiguous()
            else:
                raise ValueError(f"control batch={B_ctrl} vs prompt batch={batch_size} mismatch.")

        controlnet_cond = controlnet_cond.to(device=device, dtype=self.controlnet.dtype)
        flow_cond       = flow_cond.to(device=device, dtype=self.controlnet.dtype)

        # image size from controls unless overridden
        if height is None or width is None:
            height, width = Hc, Wc
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("height/width must be divisible by 8.")

        # -------------------------------
        # 3) Prepare timesteps & latents
        # -------------------------------
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        self._num_timesteps = len(timesteps)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,   # match UNet dtype
            device=device,
            generator=generator,
            latents=latents,
        )

        # Optional guidance-scale embedding for some UNets
        timestep_cond = None
        if getattr(self.unet.config, "time_cond_proj_dim", None) is not None:
            gs_tensor = torch.tensor(self._guidance_scale - 1).repeat(batch_size)
            timestep_cond = self.get_guidance_scale_embedding(
                gs_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # extra kwargs for scheduler (e.g., eta for DDIM); UniPC ignores generator here
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # control keep schedule (single ControlNet)
        controlnet_keep = []
        for i in range(len(timesteps)):
            keep = 1.0 - float(i / len(timesteps) < control_guidance_start[0] or (i + 1) / len(timesteps) > control_guidance_end[0])
            controlnet_keep.append(keep)

        # scale list if provided
        if isinstance(controlnet_conditioning_scale, list):
            cond_base_scale = controlnet_conditioning_scale[0]
        else:
            cond_base_scale = float(controlnet_conditioning_scale)

        # -------------------------------
        # 4) Denoising loop
        # -------------------------------
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # model inputs for this step
                if do_cfg:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    text_in = prompt_embeds                      # [2B,...]
                else:
                    latent_model_input = latents
                    text_in = prompt_embeds

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # ControlNet inputs (guess_mode follows diffusers semantics)
                if guess_mode and do_cfg:
                    control_model_input = self.scheduler.scale_model_input(latents, t)  # [B,...]
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]               # conditional only
                    control_cond_in = controlnet_cond                                   # [B,6,H,W]
                    flow_cond_in    = flow_cond                                         # [B,4,H,W]
                else:
                    control_model_input = latent_model_input                            # [2B or B,...]
                    controlnet_prompt_embeds = text_in                                  # [2B or B,...]
                    if do_cfg:
                        control_cond_in = torch.cat([controlnet_cond, controlnet_cond], dim=0)
                        flow_cond_in    = torch.cat([flow_cond, flow_cond], dim=0)
                    else:
                        control_cond_in = controlnet_cond
                        flow_cond_in    = flow_cond

                cond_scale = cond_base_scale * controlnet_keep[i]

                # ---- ControlNet forward -> residuals ----
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    sample=control_model_input,
                    timestep=t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_cond_in,
                    flow_cond=flow_cond_in,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                # If guess_mode+CFG, expand to unconditional half with zeros (official behavior)
                if guess_mode and do_cfg:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d], dim=0) for d in down_block_res_samples]
                    mid_block_res_sample   = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample], dim=0)

                # ---- UNet noise prediction ----
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_in,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self._cross_attention_kwargs,
                    down_block_additional_residuals=[d.to(dtype) for d in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype),
                    return_dict=False,
                )[0]

                # ---- Classifier-free guidance combine ----
                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

                # ---- Scheduler step (no generator arg for UniPC) ----
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # callbacks (optional)
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs if k in locals()}
                    cb_out = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = cb_out.pop("latents", latents)

                # progress bar update
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # -------------------------------
        # 5) Decode & postprocess
        # -------------------------------
        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        do_denormalize = [True] * image.shape[0] if has_nsfw_concept is None else [not x for x in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
