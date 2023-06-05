'''write a code that given the log file of the training that logs the loss_critic={} and loss_gen={} at iteration in a sequence of epochs. each line starts with percentage of the epoch and then the loss_critic and loss_gen. When the percentage goes back to 0% a new epoch starts.
 '''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(log_file, save_dir):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    loss_critic = []
    loss_gen = []
    iters = []
    epoch = 0

    epoch_loss_critic = []
    epoch_loss_gen = []
    epoch_iter = []

    #prv_perc = 0
    for line in lines:
        if line == '\n' and len(epoch_loss_critic)>0:
            loss_critic.append(epoch_loss_critic)
            loss_gen.append(epoch_loss_gen)
            iters.append(epoch_iter)
            epoch_loss_critic = []
            epoch_loss_gen = []
            epoch_iter = []
            epoch += 1
            continue
        elif line == '\n' and epoch==0:
            continue
        elif 'loss_critic' not in line:
            continue

        #cur_perc = line.split('|')[0]
        losses  = line.split('|')[2]
        
        itr = int(losses.split(' ')[1].split('/')[0])
        epoch_iter.append(itr)
        
        losses = losses.split('[')[1].split(']')[0].split(',')
        '''extract the value after loss_critic= and loss_gen='''
        for l in losses:
            if 'loss_critic=' in l:
                epoch_loss_critic.append(float(l.split('=')[1]))
            elif 'loss_gen=' in l:
                epoch_loss_gen.append(float(l.split('=')[1]))
        #print(len(epoch_loss_critic), len(epoch_loss_gen), len(epoch_iter))
        
    for e, (i,c,g) in enumerate(zip(iters, loss_critic, loss_gen)):
        plt.figure()
        plt.plot(i, c, label='loss_critic')
        plt.plot(i,g, label='loss_gen',alpha=0.5)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss_epoch{}.png'.format(e)))
        plt.close()

    '''concatenates the sublists of loss_critic and loss_gen into a single list'''
    loss_critic = [item for sublist in loss_critic for item in sublist]
    loss_gen = [item for sublist in loss_gen for item in sublist]

    plt.figure()
    plt.plot(loss_critic, label='loss_critic')
    plt.plot(loss_gen, label='loss_gen',alpha=0.5)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))

plot_loss('/home/mm3424/Research/ProGAN/Results/CelebA/PGAN_CelebA_16250169.err','.')
