# Stripped subset of the original CasCast models/latent_diffusion_model.py.
# Removes megatron_utils, wandb, horizon_metrics and all training methods.
# Only __init__ and the inference methods are kept.
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange

from models.model import basemodel
import utils.misc as utils


class latent_diffusion_model(basemodel):
    def __init__(self, logger, **params) -> None:
        super().__init__(logger, **params)
        self.logger_print_time = False
        self.data_begin_time = time.time()
        self.max_rain = 60.0
        self.diffusion_kwargs = params.get('diffusion_kwargs', {})
        self.register_buffer('scale_factor', torch.tensor(1.0))
        self.input_length = params.get("input_length", 13)
        self.pred_length  = params.get("pred_length", 12)

        ctx = params.get("context_kwargs", {})
        self.use_past_ctx   = bool(ctx.get("use_past_ctx", True))
        self.past_ctx_type  = ctx.get("past_ctx_type", "mean")
        self.past_ctx_weight = float(ctx.get("past_ctx_weight", 0.5))

        self.noise_scheduler_kwargs = self.diffusion_kwargs.get('noise_scheduler', {})
        self.noise_scheduler_type   = list(self.noise_scheduler_kwargs.keys())[0]
        _ns_cfg = self.noise_scheduler_kwargs[self.noise_scheduler_type]

        if self.noise_scheduler_type == 'DDPMScheduler':
            from src.diffusers import DDPMScheduler
            self.noise_scheduler = DDPMScheduler(**_ns_cfg)
            self.noise_scheduler.set_timesteps(_ns_cfg['num_train_timesteps'])
        elif self.noise_scheduler_type == 'DPMSolverMultistepScheduler':
            from src.diffusers import DPMSolverMultistepScheduler
            self.noise_scheduler = DPMSolverMultistepScheduler(**_ns_cfg)
            self.noise_scheduler.set_timesteps(_ns_cfg['num_train_timesteps'])
        else:
            raise NotImplementedError(f"Unknown noise scheduler: {self.noise_scheduler_type}")

        print("############# USING SAMPLER: DDIMScheduler #############")
        from src.diffusers import DDIMScheduler
        self.sample_noise_scheduler = DDIMScheduler(**_ns_cfg)
        self.sample_noise_scheduler.set_timesteps(20)

        self.noise_scale = self.noise_scheduler_kwargs.get('noise_scale', 1.0)
        self.logger.info(f'noise scale: {self.noise_scale}')

        self.predictor_ckpt_path = self.extra_params.get("predictor_checkpoint_path", None)
        if self.predictor_ckpt_path:
            self.load_checkpoint(self.predictor_ckpt_path, load_model=True,
                                 load_optimizer=False, load_scheduler=False,
                                 load_epoch=False, load_metric_best=False)

        self.autoencoder_ckpt_path = self.extra_params.get("autoencoder_checkpoint_path", None)
        if self.autoencoder_ckpt_path and os.path.exists(str(self.autoencoder_ckpt_path)):
            self.load_checkpoint(self.autoencoder_ckpt_path, load_model=True,
                                 load_optimizer=False, load_scheduler=False,
                                 load_epoch=False, load_metric_best=False)

        self.classifier_free_guidance_kwargs = self.diffusion_kwargs.get('classifier_free_guidance', {})
        self.p_uncond       = self.classifier_free_guidance_kwargs.get('p_uncond', 0.0)
        self.guidance_weight = self.classifier_free_guidance_kwargs.get('guidance_weight', 0.0)
        self.logger.info(f'INIT SCALE_FACTOR: {self.scale_factor.item()}')

    def _apply_past_context(self, z_cond, z_past):
        if (not self.use_past_ctx) or (z_past is None):
            return z_cond
        if self.past_ctx_type == "last":
            ctx = z_past[:, -1:, ...]
        else:
            ctx = z_past.mean(dim=1, keepdim=True)
        ctx = ctx.repeat(1, z_cond.shape[1], 1, 1, 1)
        return z_cond + self.past_ctx_weight * ctx

    @torch.no_grad()
    def denoise(self, template_data, cond_data, bs=1, vis=False, cfg=1.0, ensemble_member=1):
        _, T, C, H, W = template_data.shape
        cond_data = cond_data[:bs, ...]
        gen = torch.Generator(device=template_data.device)
        gen.manual_seed(0)
        latents = torch.randn((bs * ensemble_member, T, C, H, W), generator=gen,
                              device=template_data.device)
        latents = latents * self.sample_noise_scheduler.init_noise_sigma
        self.logger.info("start sampling")
        model_key = list(self.model.keys())[0]

        if float(cfg) == 1.0 and ensemble_member == 1:
            member_latents = latents[:bs, ...]
            for tstep in (tqdm(self.sample_noise_scheduler.timesteps) if vis
                          else self.sample_noise_scheduler.timesteps):
                timestep = torch.full((bs,), int(tstep), device=template_data.device,
                                      dtype=torch.long)
                latent_model_input = self.sample_noise_scheduler.scale_model_input(
                    member_latents, tstep)
                noise_pred = self.model[model_key](x=latent_model_input,
                                                   timesteps=timestep, cond=cond_data)
                member_latents = self.sample_noise_scheduler.step(
                    noise_pred, tstep, member_latents).prev_sample
            self.logger.info("end sampling")
            return member_latents

        self.logger.info(f"guidance strength: {cfg} | ens={ensemble_member}")
        cond_cat = torch.cat([cond_data, torch.zeros_like(cond_data)], dim=0)
        members = []
        for m in range(ensemble_member):
            member_latents = latents[m * bs:(m + 1) * bs, ...]
            for tstep in (tqdm(self.sample_noise_scheduler.timesteps) if vis
                          else self.sample_noise_scheduler.timesteps):
                timestep = torch.full((bs * 2,), int(tstep), device=template_data.device,
                                      dtype=torch.long)
                latent_model_input = torch.cat([member_latents, member_latents], dim=0)
                latent_model_input = self.sample_noise_scheduler.scale_model_input(
                    latent_model_input, tstep)
                noise_pred = self.model[model_key](x=latent_model_input,
                                                   timesteps=timestep, cond=cond_cat)
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + float(cfg) * (noise_pred_cond - noise_pred_uncond)
                member_latents = self.sample_noise_scheduler.step(
                    noise_pred, tstep, member_latents).prev_sample
            members.append(member_latents)
        self.logger.info("end sampling")
        return torch.stack(members, dim=1)

    @torch.no_grad()
    def decode_stage(self, z):
        z = z / self.scale_factor
        return self.model[list(self.model.keys())[1]].net.decode(z)

    @torch.no_grad()
    def init_scale_factor(self, z_tar):
        self.logger.info("### USING STD-RESCALING ###")
        x = z_tar.detach()
        local_sum   = x.sum()
        local_sumsq = (x * x).sum()
        local_n     = torch.tensor(x.numel(), device=x.device, dtype=torch.float32)
        if utils.get_world_size() > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum)
            dist.all_reduce(local_sumsq)
            dist.all_reduce(local_n)
        mean = local_sum / local_n
        std  = ((local_sumsq / local_n) - mean ** 2).clamp(min=1e-8).sqrt()
        scale_factor = (1.0 / std).clamp(min=1e-4, max=1e4)
        self.logger.info(f'scale factor: {scale_factor.item():.6f}')
        self.register_buffer('scale_factor',
                             torch.tensor(float(scale_factor), device=x.device))
