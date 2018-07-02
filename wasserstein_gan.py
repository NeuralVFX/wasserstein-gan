import torch.optim as optim
import torch
import time
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import *
from util import helpers as helper
from util import loaders as load
from models import networks as n
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


############################################################################
# Train
############################################################################

class WassGan:
    """
    Example usage if not using command line:

    params = {'dataset': 'bedroom',
                  'train_folder': 'tmp/128/train',
                  'in_channels': 3,
                  'batch_size': 128,
                  'gen_filters': 512,
                  'disc_filters': 512,
                  'z_size': 100,
                  'output_size': 64,
                  'data_perc':.003,
                  'lr_disc': 1e-4,
                  'lr_gen': 1e-4,
                  'train_epoch' : 4,
                  'gen_layers': 3,
                  'disc_layers': 4,
                  'save_root': 'lsun_test'}

    wgan = WassGan(params)
    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0
        self.preview_noise = helper.new_random_z(16, params['z_size'])

        self.transform = load.NormDenorm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.train_loader, self.data_len = load.data_load(f'data/{params["dataset"]}/{params["train_folder"]}/',
                                                          self.transform, params["batch_size"], shuffle=True,
                                                          perc=params["data_perc"], output_res=params["output_size"])

        print(f'Data Loader Initialized: {self.data_len} Images')

        self.model_dict["G"] = n.Generator(layers=params["gen_layers"], filts=params["gen_filters"],
                                           channels=params["in_channels"], z_size=params['z_size'])

        self.model_dict["D"] = n.Discriminator(layers=params["disc_layers"], filts=params["disc_filters"],
                                               channels=params["in_channels"])

        for i in self.model_dict.keys():
            self.model_dict[i].apply(helper.weights_init_normal)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        print('Networks Initialized')

        # setup optimizers #
        self.opt_dict["G"] = optim.RMSprop(self.model_dict["G"].parameters(), lr=params['lr_gen'])
        self.opt_dict["D"] = optim.RMSprop(self.model_dict["D"].parameters(), lr=params['lr_disc'])

        print('Optimizers Initialized')

        # setup history storage #
        self.losses = ['G_Loss', 'D_Loss']
        self.loss_batch_dict = {}
        self.loss_epoch_dict = {}
        self.train_hist_dict = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []

    def load_state(self, filepath):
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1
        for i in self.model_dict.keys():
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            self.opt_dict[i].load_state_dict(state['optimizers'][i])
        self.train_hist_dict = state['train_hist']

    def save_state(self, filepath):
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()
        model_state = {'iter': self.current_iter,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict}
        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def display_history(self):
        fig = plt.figure()
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def set_grad_req(self, d=True, g=True):
        for par in self.model_dict["D"].parameters():
            par.requires_grad = d
        for par in self.model_dict["G"].parameters():
            par.requires_grad = g

    def train_disc(self, data_iter):
        real = Variable(data_iter.next()).cuda()

        # clip weights in D
        for par in self.model_dict["D"].parameters():
            par.data.clamp_(-.01, .01)

        # zerp the gradient
        self.opt_dict["D"].zero_grad()

        # generate Z
        noise = helper.new_random_z(real.shape[0], self.params['z_size'])
        noise_var = Variable(noise, volatile=True)

        # generate disciminate and step
        fake = Variable(self.model_dict["G"](noise_var).data)

        d_real_result = self.model_dict["D"](real)
        d_fake_result = self.model_dict["D"](fake)

        self.loss_batch_dict['D_Loss'] = d_real_result - d_fake_result
        self.loss_batch_dict['D_Loss'].backward()
        self.opt_dict["D"].step()

    def train_gen(self):

        # zero gradient
        self.opt_dict["G"].zero_grad()

        # generate Z
        noise = helper.new_random_z(self.params["batch_size"], self.params['z_size'])
        noise_var = Variable(noise)

        # generate disciminate and step
        fake = self.model_dict["G"](noise_var)
        self.loss_batch_dict['G_Loss'] = self.model_dict["D"](fake)
        self.loss_batch_dict['G_Loss'].backward()
        self.opt_dict["G"].step()

    def train(self):
        params = self.params
        for epoch in range(params["train_epoch"]):

            # clear last epopchs losses
            for loss in self.losses:
                self.loss_epoch_dict[loss] = []

            print(f"Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
            [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}") for opt in
             self.opt_dict.keys()]

            self.model_dict["G"].train()
            self.model_dict["D"].train()

            batch_feeder = helper.BatchFeeder(self.train_loader)

            epoch_iter_count = 0
            epoch_start_time = time.time()

            with tqdm(total=self.data_len) as epoch_bar:

                while epoch_iter_count < self.data_len:

                    disc_loop_total = 100 if ((self.current_iter < 25) or (self.current_iter % 500 == 0)) else 5
                    self.set_grad_req(d=True, g=False)

                    # TRAIN DISC #
                    disc_loop_count = 0
                    while (disc_loop_count < disc_loop_total) and epoch_iter_count < self.data_len:
                        data_iter = batch_feeder.get_new()
                        self.train_disc(data_iter)

                        disc_loop_count += 1
                        epoch_iter_count += 1
                        epoch_bar.update()

                    # TRAIN GEN #
                    self.set_grad_req(d=False, g=True)
                    self.train_gen()

                    # append all losses in loss dict #
                    [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].data[0]) for loss in self.losses]
                    self.current_iter += 1

            self.current_epoch += 1

            if self.current_epoch % params['save_every'] == 0:
                helper.show_test(self.model_dict['G'], Variable(self.preview_noise),
                                 save=f'output/{params["save_root"]}_{self.current_epoch}.jpg')
                save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                print(save_str)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]
            print(f'Epoch:{self.current_epoch}, Epoch Time:{per_epoch_ptime}')
            [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]

        self.display_history()
        print('Hit End of Learning Schedule!')
