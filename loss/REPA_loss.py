import torch
import numpy as np

import torch.nn.functional as F



def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class REPALoss:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
            ):
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises

        zs_tilde = model(model_input, time_input.flatten(), **model_kwargs)


        proj_loss = 0.
        # projection loss version 1: cos_sim
        bsz = zs[0].shape[0]
        # for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
        #     for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
        #         z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
        #         z_j = torch.nn.functional.normalize(z_j, dim=-1)
        #         proj_loss += mean_flat(1 - (z_j * z_tilde_j).sum(dim=-1))
        # projection loss version 2: mse
        for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                # 计算 MSE 而不是余弦相似度
                mse_loss = torch.nn.functional.mse_loss(z_j, z_tilde_j, reduction='mean')
                proj_loss += mse_loss

        proj_loss /= (len(zs) * bsz)
        return proj_loss