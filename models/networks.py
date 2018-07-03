import torch.nn.functional as F
from torch.utils.data import *
import torch.nn as nn


############################################################################
# Re-usable blocks
############################################################################

class ConvTrans(nn.Module):
    #One block to be used as conv and transpose throughout the model
    def __init__(self, ic=4, oc=4, kernel_size=3, block_type='res', padding=None, stride=2, use_bn=True):
        super(ConvTrans, self).__init__()
        self.use_bn = use_bn

        if padding is None:
            padding = int(kernel_size // 2 // stride)

        if block_type == 'up':
            self.conv = nn.ConvTranspose2d(in_channels=ic, out_channels=oc, padding=padding, kernel_size=kernel_size,
                                           stride=stride, bias=False)
            self.relu = nn.ReLU(inplace=True)

        elif block_type == 'down':
            self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, padding=padding, kernel_size=kernel_size,
                                  stride=stride, bias=False)
            self.relu = nn.LeakyReLU(.2, inplace=True)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(oc)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


############################################################################
# Generator and Discriminator
############################################################################

class Generator(nn.Module):
    # Generator
    def __init__(self, layers=7, z_size=100, filts=1024, kernel_size=4, channels=3):
        super(Generator, self).__init__()

        operations = [ConvTrans(ic=z_size, oc=filts, kernel_size=kernel_size, padding=0, stride=1, block_type='up')]

        for a in range(layers):
            operations += [ConvTrans(ic=filts, oc=int(filts // 2), kernel_size=kernel_size, padding=1, block_type='up')]
            filts = int(filts // 2)

        operations += [ConvTrans(ic=filts, oc=filts, kernel_size=3, padding=1, stride=1, block_type='up')]
        operations += [nn.ConvTranspose2d(in_channels=filts, out_channels=channels, kernel_size=kernel_size, stride=2,
                                          bias=False,padding=1)]

        self.model = nn.Sequential(*operations)

    def forward(self, x):
        return F.tanh(self.model(x))


class Discriminator(nn.Module):
    # Discriminator grown in reverse
    def __init__(self, channels=3, filts=512, kernel_size=4, layers=5):
        super(Discriminator, self).__init__()

        operations = [nn.Conv2d(in_channels=filts, out_channels=1, padding=0, kernel_size=kernel_size, stride=1)]

        for a in range(layers):
            operations += [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, block_type='down')]
            filts = int(filts // 2)

        operations += [ConvTrans(ic=filts, oc=filts, kernel_size=3, stride=1, block_type='down')]
        operations += [ConvTrans(ic=channels, oc=filts, kernel_size=kernel_size, use_bn=False, block_type='down')]
        operations.reverse()

        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        x = self.operations(x)
        return x.mean(0).view(1)
