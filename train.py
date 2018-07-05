#!/usr/bin/env python
import argparse
from wasserstein_gan import WassGan


parser = argparse.ArgumentParser()

parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--dataset', nargs='?', default='bedroom', type=str)
parser.add_argument('--train_folder', nargs='?', default='train', type=str)
parser.add_argument('--in_channels', nargs='?', default=3, type=int)
parser.add_argument('--batch_size', nargs='?', default=128, type=int)
parser.add_argument('--gen_filters', nargs='?', default=512, type=int)
parser.add_argument('--disc_filters', nargs='?', default=512, type=int)
parser.add_argument('--z_size', nargs='?', default=100, type=int)
parser.add_argument('--output_size', nargs='?', default=64, type=int)
parser.add_argument('--data_perc', nargs='?', default=.003, type=float)
parser.add_argument('--lr_disc', nargs='?', default=1e-4, type=float)
parser.add_argument('--lr_gen', nargs='?', default=1e-4, type=float)
parser.add_argument('--train_epoch', nargs='?', default=300, type=int)
parser.add_argument('--gen_layers', nargs='?', default=3, type=int)
parser.add_argument('--disc_layers', nargs='?', default=3, type=int)
parser.add_argument('--save_every', nargs='?', default=3, type=int)
parser.add_argument('--save_root', nargs='?', default='lsun_test', type=str)
parser.add_argument('--load_state', nargs='?', type=str)

params = vars(parser.parse_args())

# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    wgan = WassGan(params)
    if params['load_state']:
        wgan.load_state(params['load_state'])
    else:
        print('Starting From Scratch')
    wgan.train()
