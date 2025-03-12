import torch
import torch.distributed as dist

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

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

create_npz_from_sample_folder("/mnt/ssd/dataset/image/imagenet-1k/generated-result/modified-FA-5-lambda1-nocfg")