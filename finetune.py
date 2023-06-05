""" Training of ProGAN using WGAN-GP loss"""

import logging
import os
import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    createdir,
    generate_examples,
    generate_fixed_examples,
    plot_loss_to_tensorboard,
    plot_images_to_tensorboard,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
from scipy.stats import truncnorm

#from utils_FID.inception_score import _init_inception
#from utils_FID.fid_score import create_inception_graph, check_or_download_inception

torch.backends.cudnn.benchmarks = True

_logger = logging.getLogger(__name__)

#import sys
#save_dir = sys.argv[1]
#load = sys.argv[2]
#epochs = sys.argv[3]

#try:
#    gen_checkpoint = sys.argv[3] # checkpoint to load: if checkpoint has field "modify", then modify the model architecture accordingly
#    crit_chekcpoint = sys.argv[4]
#    assert(os.path.exists(gen_checkpoint))
#    assert(os.path.exists(crit_chekcpoint))
#except:
#    gen_checkpoint = config.GEN_CHECKPOINT
#    crit_chekcpoint = config.CRIT_CHECKPOINT




def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_img_step,
    tensorboard_loss_step,
    writer,
    scaler_gen,
    scaler_critic,
    epoch,
    save_dir
):
    # Generated Image parameters
    truncation=10
    n = 100
    num_batches = len(loader)
    #fixed_noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(n, config.Z_DIM, 1, 1)), device=config.DEVICE, dtype=torch.float32)
    fixed_noise = torch.randn(n, config.Z_DIM, 1, 1).to(config.DEVICE)
    #
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        for i in range(config.MINIBATCH_REPEATS):
            # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
            # which is equivalent to minimizing the negative of the expression
            noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)

            # Train Critic:
            with torch.cuda.amp.autocast():
                fake = gen(noise, alpha, step)
                critic_real = critic(real, alpha, step)
                critic_fake = critic(fake.detach(), alpha, step)
                gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )
            if torch.isnan(loss_critic):
                _logger.error("critic loss is NaN (Before scaling) at step %d, epoch %d, batch idx %d ", step, epoch, batch_idx)

            opt_critic.zero_grad()
            scaler_critic.scale(loss_critic).backward()
            if torch.isnan(loss_critic):
                _logger.error("critic loss is NaN (After scaling) at step %d, epoch %d, batch idx %d ", step, epoch, batch_idx)
            scaler_critic.step(opt_critic)
            scaler_critic.update()
            if torch.isnan(loss_critic):
                _logger.error("critic loss is NaN (After unscaling) at step %d, epoch %d, batch idx %d ", step, epoch, batch_idx)

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            with torch.cuda.amp.autocast():
                gen_fake = critic(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)


            if torch.isnan(loss_gen):
                _logger.error("generator loss is NaN (Before scaling) at step %d, epoch %d, batch idx %d ", step, epoch, batch_idx)

            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            if torch.isnan(loss_gen):
                _logger.error("generator loss is NaN (After scaling) at step %d, epoch %d, batch idx %d ", step, epoch, batch_idx)
            scaler_gen.step(opt_gen)
            scaler_gen.update()
            if torch.isnan(loss_gen):
                _logger.error("generator loss is NaN (After unscaling) at step %d, epoch %d, batch idx %d ", step, epoch, batch_idx)

            # Update alpha and ensure less than 1
            alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset) * config.MINIBATCH_REPEATS
            )
            
            alpha = min(alpha, 1)
        plot_loss_to_tensorboard(writer, loss_critic.item(), loss_gen.item(), tensorboard_loss_step)
        tensorboard_loss_step += 1
        
        if batch_idx % 500 == 0 or (batch_idx == num_batches - 1):
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_images_to_tensorboard(
                writer,
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_img_step,
            )
            tensorboard_img_step += 1
            
            save_image(fixed_fakes, save_dir + f"/Images/img_{step}_{epoch}_{batch_idx}.png")
            # generated images save directory
            dir_ = save_dir + f"/Generated_Images_{step}_{epoch}_{batch_idx}"
            createdir(dir_)
            generate_fixed_examples(gen, config.MAX_IMG_SIZE_IDX, alpha, truncation=10, n=config.n, dir_=dir_, noise=config.FIXED_NOISE2)

        loop.set_postfix(
            gp=gp.item(),
            loss_gen=loss_gen.item(),
            loss_critic=loss_critic.item(),
        )
        _logger.info('Step: %d, Epoch: %d, Batch: %d/%d, GP:%.4g, loss_critic:%.4g, loss_gen=%.4g ', step, epoch, batch_idx, num_batches, gp.item(), loss_critic.item(), loss_gen.item())

    return tensorboard_img_step, tensorboard_loss_step, alpha


def finetune(gen, critic, opt_gen, opt_critic, epochs, save_dir ):
    
    _logger.info("Starting finetuning...")
    
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(save_dir+ "/logs/gan1")

    gen.train()
    critic.train()

    tensorboard_img_step, tensorboard_loss_step = 0, 0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    
    alpha = 1  # start with very low alpha
    loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
    print(f"Current image size: {4 * 2 ** step}")
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        tensorboard_img_step, tensorboard_loss_step, alpha = train_fn(
            critic,
            gen,
            loader,
            dataset,
            step,
            alpha,
            opt_critic,
            opt_gen,
            tensorboard_img_step,
            tensorboard_loss_step,
            writer,
            scaler_gen,
            scaler_critic,
            epoch,
            save_dir
        )
        if epoch<len(epochs)-1:
            _logger.info("Saving checkpoint...")
            save_checkpoint(gen, opt_gen, filename=save_dir+ f"/generator_{epoch}.pth")
            save_checkpoint(critic, opt_critic, filename=save_dir+ f"/critic_{epoch}.pth")

    _logger.info("Saving checkpoint...")
    save_checkpoint(gen, opt_gen, filename=save_dir+ "/generator.pth")
    save_checkpoint(critic, opt_critic, filename=save_dir+ "/critic.pth")

