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


from utils_FID.inception_score import _init_inception
from utils_FID.fid_score import create_inception_graph, check_or_download_inception

torch.backends.cudnn.benchmarks = True



def main():

    createdir("Results")
    createdir("Results/" + config.DATASET_NAME)
    createdir("Results/" + config.DATASET_NAME + "/Generated_Images")

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    #critic = Discriminator(
    #    config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    #).to(config.DEVICE)

    # initialize optimizers and scalers for FP16 training
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )

    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    #load_checkpoint(
    #    config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
    #)

    gen.eval()
    #critic.eval()
    #generate_examples(gen, config.MAX_IMG_SIZE_IDX, truncation=10, n=50000)
     # set TensorFlow environment for evaluation (calculate IS and FID)
    _init_inception()
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)

    fid_stat = 'fid_stat/fid_stats_celebA_train.npz'
    inception_score, std, fid_score = validate(fid_stat, gen, steps=config.MAX_IMG_SIZE_IDX)
    print(f'Inception score mean: {inception_score}, Inception score std: {std}, '
                f'FID score: {fid_score}.')

if __name__ == "__main__":
    main()
