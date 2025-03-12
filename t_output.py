# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from fsspec.registry import default

from models.flex_alignment import FA_models
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model


torch.set_grad_enabled(False)

device = torch.device('cpu')

# Load model:
block_kwargs = {"fused_attn": False, "qk_norm": False}
latent_size = 256 // 8
# SiT_models
model = FA_models["FA-L/2"](
    input_size=latent_size,
    num_classes=1000,
    use_cfg = True,
    z_dims = [768],
    encoder_depth=8,
    **block_kwargs,
).to(device)
# Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
ckpt_path = "/mnt/ssd/model/image/repa/FA/5-b-enc8-lambda0.5/checkpoints/0400000.pt"
state_dict = torch.load(ckpt_path, map_location=f'{device}',weights_only=False)['ema']

model.load_state_dict(state_dict)
model.eval()  # important!
z = torch.randn(1, model.in_channels, 256//8, 256//8, device=device)
y = torch.randint(0, 1000, (1,), device=device)

# Sample images:
sampling_kwargs = dict(
    model=model,
    latents=z,
    y=y,
    num_steps=50,
    heun=False,
    cfg_scale=0,
    guidance_low=0.,
    guidance_high=1.,
    path_type="linear",
)
with torch.no_grad():
    samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)




