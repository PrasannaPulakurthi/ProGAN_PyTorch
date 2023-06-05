import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
import config
from torchvision.utils import save_image
from scipy.stats import truncnorm
#from utils_FID.inception_score import get_inception_score
#from utils_FID.fid_score import calculate_fid_given_paths
#from imageio import imsave
import logging
import typing
import sys
import zipfile
import glob


_logger = logging.getLogger(__name__)

# Print losses occasionally and print to tensorboard
def plot_to_tensorboard(
    writer, loss_critic, loss_gen, real, fake, tensorboard_step
):
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)

def plot_loss_to_tensorboard(writer, loss_critic, loss_gen, tensorboard_step):
    writer.add_scalar("Loss Generator", loss_gen, global_step=tensorboard_step)
    writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

def plot_images_to_tensorboard(writer, real, fake, tensorboard_step):
    with torch.no_grad():
        # take out (up to) 8 examples to plot
        img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
        writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
        writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean(((gradient_norm/config.GAMMA - 1) ** 2))
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    print(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(gen, steps, truncation=0.7, n=100):
    """
    Tried using truncation trick here but not sure it actually helped anything, you can
    remove it if you like and just sample from torch.randn
    """
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)), device=config.DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img*0.5+0.5, 'Results/'+ config.DATASET_NAME + f"/Generated_Images/img_{i}.png")
            print(i)
    gen.train()
def generate_fixed_examples(gen, steps, alpha, truncation=0.7, n=100, dir_='', noise=None):
    """
    Tried using truncation trick here but not sure it actually helped anything, you can
    remove it if you like and just sample from torch.randn
    """
    if dir_ == '':
        raise Exception('Please specify a directory to save the images')
    if noise is None:
        #noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(n, config.Z_DIM, 1, 1)), device=config.DEVICE, dtype=torch.float32)
        noise = torch.randn(n, config.Z_DIM, 1, 1).to(config.DEVICE)

    gen.eval()
    #alpha = 1.0
    with torch.no_grad():
        img = gen(noise, alpha, steps)
    save_image(img*0.5+0.5, dir_ + f"/img.png")
    for i in range(n):
        with torch.no_grad():
            noise_i = noise[i].unsqueeze(0)
            img = gen(noise_i, alpha, steps)
            save_image(img*0.5+0.5, dir_ + f"/img_{i}.png")
    gen.train()

def createdir(path):
    # Check whether the specified path exists or not
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory created at:"+path)
    else:
        print("The directory already exits at:"+path)

class LogExceptionHook(object):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(self, exc_type, exc_value, traceback):
        self.logger.exception("Uncaught exception", exc_info=(
            exc_type, exc_value, traceback))

def init_logger(filename: str = None) -> logging.Logger:
        _logger = logging.getLogger(__name__)
        for handler in _logger.handlers[:]:  # make a copy of the list
            _logger.removeHandler(handler)
        FORMAT = "%(asctime)s %(levelname)-8s: %(message)s"    
        formatter = logging.Formatter(FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers = [console_handler]
        if filename is not None:
            file_handler = logging.FileHandler(filename, mode="w")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO"))

        logging.basicConfig(
            level=level,
            handlers=handlers
        )

        sys.excepthook = LogExceptionHook(_logger)

def create_code_snapshot(name: str,
                         include_suffix: typing.List[str],
                         source_directory: str,
                         store_directory: str) -> None:
    
    if store_directory is None:
        return
    with zipfile.ZipFile(os.path.join(store_directory, "{}.zip".format(name)), "w") as f:
        for suffix in include_suffix:
            for file in glob.glob(os.path.join(source_directory, "**", "*{}".format(suffix)), recursive=True):
                f.write(file, os.path.join(name, file))

"""
def validate(fid_stat, fid_bugger_dir, gen_net: nn.Module, steps):

    # eval mode
    gen_net = gen_net.eval()

    # get fid and inception score
    #fid_buffer_dir = "Results/"+ config.DATASET_NAME + f"/Generated_Images"
    os.makedirs(fid_buffer_dir, exist_ok=True)

    eval_iter = config.NUM_EVAL_IMGS // config.EVAL_BATCH_SIZE
    img_list = list()
    alpha = 1.0
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        noise = torch.randn(config.EVAL_BATCH_SIZE, config.Z_DIM, 1, 1).to(config.DEVICE)
        # generate a batch of images
        gen_imgs = gen_net(noise, alpha, steps)
        gen_imgs = gen_imgs.mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'img_{iter_idx}_{img_idx}.png')
            imsave(file_name, img)
        img_list.extend(list(gen_imgs))

    # get inception score
    print('=> calculate inception score')
    mean, std = get_inception_score(img_list)

    # get fid score
    print('=> calculate fid score')
    fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)
    
    return mean, std , fid_score
"""