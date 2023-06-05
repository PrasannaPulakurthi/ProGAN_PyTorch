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
    generate_fixed_examples,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
from td_utils import *
from finetune import * 

torch.backends.cudnn.benchmarks = True

import sys
init = sys.argv[1]  # True if we want to initialize the folder with checkpoints & config files, False if we want to compress & fine-tune
if init not in ['True', 'False']:
    raise ValueError('init should be either True or False')
init = True if init == 'True' else False
print(init)
layer_names = sys.argv[2]
layers = layer_names.split(' ')
print(layers)

rank = int(sys.argv[3])
epochs = int(sys.argv[4])
dir_ = sys.argv[5]
#only_evaluate = sys.argv[6]

def main():

    createdir("Results")
    createdir("Results/" + config.DATASET_NAME)
    createdir("Results/" + config.DATASET_NAME + "/Generated_Images")
    createdir(dir_)
    save_dir = dir_

    # initialize gen and disc, note: discriminator should be called critic,
    # according to WGAN paper (since it no longer outputs between [0, 1])
    # but really who cares..
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    original_gen_size = sum([p.numel() for p in gen.parameters()])

    
    if init:
        # just save the initial models & config files to the base directory
        save_dir = save_dir+'/'+'Base'

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

        

        conf = Config(save_dir=save_dir, modify=None, step=0)
        conf.save_config()
        save_checkpoint(gen, opt_gen, filename=save_dir+'/generator.pth')
        save_checkpoint(critic, opt_critic, filename=save_dir+'/critic.pth')

    else:
        save_dir = save_dir+'/'+layer_names+'-'+str(rank)
        createdir(save_dir)
        createdir(save_dir + f"/Images")

        in_conf = Config(dir_)
        in_conf.load_config(dir_+'/config.json')
        step = in_conf.step
        if in_conf.modify is not None:
            print('HERE')
            # modity_arch according to the config file
            gen = modify_gen(gen, in_conf.modify)

        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
        opt_critic = optim.Adam(
            critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
        )
        print(dir_)
        load_checkpoint(
            dir_+'/generator.pth', gen, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            dir_+'/critic.pth', critic, opt_critic, config.LEARNING_RATE
        )

        loaded_gen_size = sum([p.numel() for p in gen.parameters()])
        
        gen.eval()
        critic.eval()
        print('Model architecture before decomposition: ')
        print(gen)
        print('##################################')
        for layer_name in layers:
            approx_error = decompose_and_replace_conv_layer_by_name(gen, layer_name, rank=rank, freeze=False)
            print('Approximation error for layer ', layer_name, ' is ', approx_error)

        compressed_size = sum([p.numel() for p in gen.parameters()])
        print('Number of parameters reduced by: ', 100*(original_gen_size - compressed_size)/original_gen_size, '%')

        print('Original size of the Generator: ', original_gen_size)
        print('Size of the loaded Generator: ', loaded_gen_size)
        print('Size of the compressed Generator: ', compressed_size)
        print('##################################')

        conf_modify = in_conf.modify
        #print('conf_modify is ', conf_modify)
        if conf_modify is None:
            conf_modify = {}
        conf_modify[layer_name] = rank
        conf = Config(save_dir=save_dir, modify=conf_modify, step=step+1)
        conf.save_config()

        print('Model architecture after decomposition: ')
        print(gen)
        print('##################################')

        imgdir = save_dir+f"/Generated_Images_before_finetune"
        createdir(imgdir)
        generate_fixed_examples(gen, config.MAX_IMG_SIZE_IDX, alpha=1, truncation=10, n=100, dir_=imgdir)
        finetune(gen, critic, opt_gen, opt_critic, epochs, save_dir)





    #gen.train()
    #critic.train()
    
    #conv_layers_info = get_conv2d_layers_info(gen)
    #print('Convolution layers of the Generator: ')
    #for c,v in conv_layers_info.items():
    #    print('Layer "',c,'", size: ', v)
    #print('##################################')

    #gen.eval()
    #critic.eval()
    #approx_error = decompose_and_replace_conv_layer_by_name(gen, layer_name, rank=rank, freeze=False, device='cpu')
    


    


    #generate_examples(gen, config.MAX_IMG_SIZE_IDX, truncation=10, n=50000)

if __name__ == "__main__":
    main()
