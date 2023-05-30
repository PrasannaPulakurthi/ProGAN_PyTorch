""" Training of ProGAN using WGAN-GP loss"""

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    createdir,
    generate_examples,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
from td_utils import *

torch.backends.cudnn.benchmarks = True

import sys
layer_name = sys.argv[1]
save_dir = sys.argv[2]


def main():

    createdir("Results")
    createdir("Results/" + config.DATASET_NAME)
    createdir("Results/" + config.DATASET_NAME + "/Generated_Images")
    createdir(save_dir)
    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )

    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        #'/home/mm3424/Research/ProGAN/Backup_Results/CelebA/generator_3.pth', gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
    )

    #gen.train()
    #critic.train()
    
    conv_layers_info = get_conv2d_layers_info(gen)
    print('Convolution layers of the Generator: ')
    for c,v in conv_layers_info.items():
        print('Layer "',c,'", size: ', v)

    ranks, approximations = get_conv2d_layer_approximation_vs_rank(gen, layer_name, max_rank = None, decompose_type='cp', save_fig=True, save_path=save_dir)
    print(f"Ranks: {ranks}")
    print(f"Approximation errors {approximations}")

    #generate_examples(gen, config.MAX_IMG_SIZE_IDX, truncation=10, n=50000)

if __name__ == "__main__":
    main()
