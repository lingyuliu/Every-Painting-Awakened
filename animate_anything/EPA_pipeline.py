import torch
from einops import rearrange, repeat
import numpy as np
from models.pipeline import LatentToVideoPipeline


class PaintingAwakened_Pipeline(LatentToVideoPipeline):

    @torch.no_grad()
    def ScoreDistillationSampling(self, image_latent, prompt, num_inference_steps, p_ni_vsds, mask, motion, guidance_scale, diffusion_scheduler):

        callback = None
        callback_steps = 1
        num_warmup_steps = 0
        num_frames = 16
        step = 25
        timesteps = diffusion_scheduler.timesteps[len(diffusion_scheduler.timesteps) - step:]
        do_classifier_free_guidance = guidance_scale > 1.0

        image_latents = repeat(image_latent, 'b c 1 h w -> b c f h w', f=num_frames)

        device = image_latents.device

        # Encode input prompt
        text_encoder_lora_scale = (
            None
        )
        num_images_per_prompt = 1
        negative_prompt = None
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )

        constant_noise = torch.randn_like(image_latents, dtype=image_latents.dtype, device=device)

        # Video Score Distillation Sampling
        tau_ni_vsds = int(num_inference_steps * p_ni_vsds) + 1

        ni_vsds_latents = image_latents.clone().detach()
        first_half = np.linspace(10, 1, int(tau_ni_vsds // 2)).tolist()
        second_half = [1] * (int(tau_ni_vsds // 2))
        step_size_list = first_half + second_half

        uncondition_latent = image_latent
        condition_latent = torch.cat(
            [uncondition_latent, image_latent]) if do_classifier_free_guidance else image_latent
        with self.progress_bar(total=tau_ni_vsds) as progress_bar:
            for i, t in enumerate(timesteps[:tau_ni_vsds]):

                tt = torch.tensor([t] * ni_vsds_latents.shape[0], device=device)
                noise_latents = self.scheduler.add_noise(ni_vsds_latents, constant_noise, tt)

                latent_model_input = torch.cat(
                    [noise_latents] * 2) if do_classifier_free_guidance else noise_latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if motion is not None:
                    motion = torch.tensor(motion, device=device)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    condition_latent=condition_latent,
                    mask=mask,
                    motion=motion
                ).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                w = (1 - self.scheduler.alphas_cumprod[t])  # Score Distillation Sampling w(t)
                grad = w * (noise_pred - constant_noise)

                step_size = step_size_list[i]
                ni_vsds_latents = ni_vsds_latents - step_size * grad

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, image_latents)
        return ni_vsds_latents.clone()

    @staticmethod
    def DDPM_forward_timesteps_slerp(x0, x1, step, num_frames, scheduler):
        '''larger step -> smaller t -> smaller alphas[t:] -> smaller xt -> smaller x0'''

        device = x0.device
        timesteps = scheduler.timesteps[len(scheduler.timesteps) - step:]
        t = timesteps[0]

        def generate_slerp_frames_16(A, B, frames):

            def slerp(p0, p1, fract_mixing: float):
                p0 = p0.to(dtype=torch.half)
                p1 = p1.to(dtype=torch.half)

                p0 = p0.double()
                p1 = p1.double()
                norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
                epsilon = 1e-7
                dot = torch.sum(p0 * p1) / norm
                dot = dot.clamp(-1 + epsilon, 1 - epsilon)

                theta_0 = torch.arccos(dot)
                sin_theta_0 = torch.sin(theta_0)
                theta_t = theta_0 * fract_mixing
                s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
                s1 = torch.sin(theta_t) / sin_theta_0
                interp = p0 * s0 + p1 * s1

                interp = interp.to(dtype=torch.half)
                return interp

            b, c, _, h, w = A.shape
            interpolated_tensor = torch.zeros((b, c, frames, h, w), dtype=A.dtype, device=A.device)
            timesteps = torch.linspace(0, 1, frames)
            for i, t in enumerate(timesteps):
                if i == 0:
                    interpolated_tensor[:, :, i, :, :] = A[:, :, i, :, :]
                elif i == frames - 1:
                    interpolated_tensor[:, :, i, :, :] = B[:, :, i, :, :]
                else:
                    interpolated_tensor[:, :, i, :, :] = slerp(A[:, :, i, :, :], B[:, :, i, :, :], t)

            return interpolated_tensor

        xt = generate_slerp_frames_16(x0, x1, num_frames)

        noise = torch.randn(xt.shape, dtype=xt.dtype, device=device)

        t = torch.tensor([t] * xt.shape[0], device=device)
        xt = scheduler.add_noise(xt, noise, t)
        return xt, timesteps
