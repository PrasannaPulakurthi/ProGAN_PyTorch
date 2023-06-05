""" Training of ProGAN using WGAN-GP loss"""

import os
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    createdir,
    generate_examples,
    validate,
)
from math import log2
from tqdm import tqdm
import config
from imageio import imsave

torch.backends.cudnn.benchmarks = True


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), 
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = 1
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset

def main():
    fid_buffer_dir = "Results/"+ config.DATASET_NAME + f"/Real_Images"
    os.makedirs(fid_buffer_dir, exist_ok=True)
    step = 4
    loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
    loop = tqdm(loader, leave=True)
    i=1
    for batch_idx, (real, _) in enumerate(loop):
        file_name = os.path.join(fid_buffer_dir, f'img_{i}.png')
        real = real.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        imsave(file_name, real[0])
        i=i+1

if __name__ == "__main__": 
    main()