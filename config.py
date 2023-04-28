import torch
from math import log2

DATASET_NAME = 'celebA'
MAX_IMG_SIZE = 64
MAX_IMG_SIZE_IDX = int(log2(MAX_IMG_SIZE)-2)
START_TRAIN_AT_IMG_SIZE = 4
DATASET = '../../Datasets/' + DATASET_NAME
CHECKPOINT_GEN = 'Results/' + DATASET_NAME + "/generator.pth"
CHECKPOINT_CRITIC = 'Results/' + DATASET_NAME + "/critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
# BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
BATCH_SIZES = [32, 32, 32, 16, 16]
CHANNELS_IMG = 3
Z_DIM = 512  # should be 512 in original paper
IN_CHANNELS = 512  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
GAMMA = 10
PROGRESSIVE_EPOCHS = [8] * len(BATCH_SIZES)
MINIBATCH_REPEATS = 4
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 9