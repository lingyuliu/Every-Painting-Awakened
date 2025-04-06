import argparse
import math
import os

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.utils.checkpoint
import torchvision.transforms as T
import numpy as np

from PIL import Image
from accelerate.utils import set_seed

from diffusers import DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor

from einops import rearrange, repeat
import imageio

from EPA_pipeline import PaintingAwakened_Pipeline

from utils.common import tensor_to_vae_latent
from train import load_primary_models,freeze_models,handle_memory_attention,cast_to_gpu_and_type


def eval(pipeline, vae_processor, validation_data, out_file, forward_t=25, preview=True):
    vae = pipeline.vae
    diffusion_scheduler = pipeline.scheduler
    device = vae.device
    dtype = vae.dtype

    prompt = validation_data.prompt
    pimg = Image.open(validation_data.prompt_image)
    if pimg.mode == "RGBA":
        pimg = pimg.convert("RGB")
    width, height = pimg.size
    scale = math.sqrt(width * height / (validation_data.height * validation_data.width))
    validation_data.height = round(height / scale / 8) * 8
    validation_data.width = round(width / scale / 8) * 8
    input_image = vae_processor.preprocess(pimg, validation_data.height, validation_data.width)
    input_image = input_image.unsqueeze(0).to(dtype).to(device)
    input_image_latents = tensor_to_vae_latent(input_image, vae)

    proxy_img = Image.open(validation_data.proxy_image)
    if proxy_img.mode == "RGBA":
        proxy_img = proxy_img.convert("RGB")
    proxy_img = proxy_img.resize((width, height))
    proxy_image = vae_processor.preprocess(proxy_img, validation_data.height, validation_data.width)
    proxy_image = proxy_image.unsqueeze(0).to(dtype).to(device)
    proxy_image_latents = tensor_to_vae_latent(proxy_image, vae)

    motion_strength = validation_data.get("strength", 8)
    np_mask = np.ones([validation_data.height, validation_data.width], dtype=np.uint8) * 255
    mask = T.ToTensor()(np_mask).to(dtype).to(device)
    b, c, _, h, w = input_image_latents.shape
    mask = T.Resize([h, w], antialias=False)(mask)
    mask = rearrange(mask, 'b h w -> b 1 1 h w')

    with torch.no_grad():
        input_image_latents_new = pipeline.ScoreDistillationSampling(image_latent=input_image_latents,
                                                                     prompt=prompt,
                                                                     num_inference_steps=validation_data.num_inference_steps,
                                                                     p_ni_vsds=0.6,
                                                                     mask=mask,
                                                                     motion=[motion_strength],
                                                                     guidance_scale=validation_data.guidance_scale,
                                                                     diffusion_scheduler=diffusion_scheduler)
        proxy_image_latents_new = pipeline.ScoreDistillationSampling(image_latent=proxy_image_latents,
                                                                      prompt=prompt,
                                                                      num_inference_steps=validation_data.num_inference_steps,
                                                                      p_ni_vsds=0.6,
                                                                      mask=mask,
                                                                      motion=[motion_strength],
                                                                      guidance_scale=validation_data.guidance_scale,
                                                                      diffusion_scheduler=diffusion_scheduler)

    initial_latents, timesteps = pipeline.DDPM_forward_timesteps_slerp(input_image_latents_new, proxy_image_latents_new, forward_t,
                                                                   validation_data.num_frames, diffusion_scheduler)

    with torch.no_grad():
        video_frames, video_latents = pipeline(
            prompt=prompt,
            latents=initial_latents,
            width=validation_data.width,
            height=validation_data.height,
            num_frames=validation_data.num_frames,
            num_inference_steps=validation_data.num_inference_steps,
            guidance_scale=validation_data.guidance_scale,
            condition_latent=input_image_latents,
            mask=mask,
            motion=[motion_strength],
            return_dict=False,
            timesteps=timesteps,
        )
    if preview:
        fps = validation_data.get('fps', 8)
        imageio.mimwrite(out_file, video_frames, duration=int(1000 / fps), loop=0)
        imageio.mimwrite(out_file.replace('gif', '.mp4'), video_frames, fps=fps)
    print(f"save file {out_file}")

    del pipeline
    torch.cuda.empty_cache()


def batch_eval(unet, text_encoder, vae, vae_processor, pretrained_model_path,
               validation_data, output_dir, preview):
    device = vae.device
    dtype = vae.dtype
    unet.eval()
    text_encoder.eval()
    pipeline = PaintingAwakened_Pipeline.from_pretrained(
        pretrained_model_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet
    )

    diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    diffusion_scheduler.set_timesteps(validation_data.num_inference_steps, device=device)
    pipeline.scheduler = diffusion_scheduler

    name = os.path.basename(validation_data.prompt_image)
    os.makedirs(output_dir, exist_ok=True)
    out_file = f"{output_dir}/{name[:-4]}_{validation_data.prompt}.gif"
    eval(pipeline, vae_processor, validation_data, out_file, forward_t=validation_data.num_inference_steps, preview=preview)


def main_eval(
        pretrained_model_path: str,
        validation_data: Dict,
        enable_xformers_memory_efficient_attention: bool = True,
        enable_torch_2_attn: bool = False,
        seed: Optional[int] = None,
        motion_mask=False,
        motion_strength=False,
        **kwargs
):
    if seed is not None:
        set_seed(seed)
    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path, motion_strength=motion_strength)
    vae_processor = VaeImageProcessor()
    # Freeze any necessary models
    freeze_models([vae, text_encoder, unet])

    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.half

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, unet, vae]
    cast_to_gpu_and_type(models_to_cast, torch.device("cuda"), weight_dtype)
    batch_eval(unet, text_encoder, vae, vae_processor, pretrained_model_path,
               validation_data, "./output", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.eval = True
    args_dict = OmegaConf.load(args.config)
    cli_dict = OmegaConf.from_dotlist(args.rest)
    args_dict = OmegaConf.merge(args_dict, cli_dict)
    main_eval(**args_dict)
