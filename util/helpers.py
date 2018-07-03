import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import *

############################################################################
# Helper Utilities
############################################################################


def mft(tensor):
    # Return mean float tensor
    return torch.mean(torch.FloatTensor(tensor))


def normalize_img(x, cpu=False):
    # Reverse Image Normalization
    if cpu:
        x = x.cpu().data
    return (x.numpy().transpose(1, 2, 0) + 1) / 2


def weights_init_normal(m):
    # Set initial state of weights
    classname = m.__class__.__name__
    if 'ConvTrans' == classname:
        pass
    elif 'Conv2d' in classname or 'Linear' in classname or 'ConvTrans' in classname:
        nn.init.normal(m.weight.data, 0, .02)


def new_random_z(bs, z, seed = False):
    # Creates Z vector of normally distributed noise
    if seed:
        torch.manual_seed(seed)
    z = torch.FloatTensor(bs, z, 1, 1).normal_(0, 1).cuda()
    return z


class BatchFeeder:
    # Abstracting the fetching of batches from the data loader
    def refresh(self):
        self.data_iter = iter(self.train_loader)
        self.batch_len = len(self.train_loader)
        self.data_iter_count = 0

    def __init__(self, dataloader):
        self.train_loader = dataloader
        self.original_len = len(self.train_loader)
        self.refresh()

    def get_new(self):
        if self.data_iter_count < self.batch_len:
            self.refresh()
        batch = self.data_iter
        self.data_iter_count += 1
        return batch

############################################################################
# Display Images
############################################################################

def show_test(gen, z, save):
    # Generate samples from z vector, show and also save
    gen.eval()
    results = gen(z)
    gen.train()
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(normalize_img(results[i], cpu=True))
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)


def show_samples(results):
    # Show samples, used to show raw samples from dataset
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(normalize_img(results[i], cpu=True))
    plt.show()
    plt.close(fig)
