""" Training of ProGAN using WGAN-GP loss"""

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
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config

torch.backends.cudnn.benchmarks = True

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
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
    epoch,
):
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

            opt_critic.zero_grad()
            scaler_critic.scale(loss_critic).backward()
            scaler_critic.step(opt_critic)
            scaler_critic.update()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            with torch.cuda.amp.autocast():
                gen_fake = critic(fake, alpha, step)
                loss_gen = -torch.mean(gen_fake)

            opt_gen.zero_grad()
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            # Update alpha and ensure less than 1
            alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset) * config.MINIBATCH_REPEATS
            )
            alpha = min(alpha, 1)
        
        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1
            
            save_image(fixed_fakes, 'Results/'+ config.DATASET_NAME + f"/Images/img_{step}_{epoch}_{batch_idx}.png")

        loop.set_postfix(
            gp=gp.item(),
            loss_gen=loss_gen.item(),
            loss_critic=loss_critic.item(),
        )

    return tensorboard_step, alpha


def finetune(gen, critic, opt_gen, opt_ciritc, epochs, save_dir ):
    
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # for tensorboard plotting
    writer = SummaryWriter(save_dir+ "/logs/gan1")

    gen.train()
    critic.train()

    tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    
    alpha = 1e-5  # start with very low alpha
    loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
    print(f"Current image size: {4 * 2 ** step}")
    
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        tensorboard_step, alpha = train_fn(
            critic,
            gen,
            loader,
            dataset,
            step,
            alpha,
            opt_critic,
            opt_gen,
            tensorboard_step,
            writer,
            scaler_gen,
            scaler_critic,
            epoch,
        )


    save_checkpoint(gen, opt_gen, filename=save_dir+ "/generator.pth")
    save_checkpoint(critic, opt_critic, filename=save_dir+ "/critic.pth")

