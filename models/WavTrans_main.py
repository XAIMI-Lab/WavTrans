import os
from math import log10
import scipy.io as sio
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from skimage.measure import compare_ssim as ssim
import math
import numbers
import torch.nn.functional as F
from networks import get_generator_guide
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, get_nonlinearity, DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2_net, complex_abs_eval
import numpy as np
import pdb

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, weight_size, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(weight_size, numbers.Number):
            weight_size = [weight_size] * dim
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        weight_kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in weight_size])
        for size, std, mgrid in zip(weight_size, sigma, meshgrids):
            mean = (size - 1) / 2
            weight_kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        weight_kernel = weight_kernel / torch.sum(weight_kernel)

        kernel = torch.zeros(kernel_size)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        weight_kernel = weight_kernel.view(1, 1, *weight_kernel.size())
        weight_kernel = weight_kernel.repeat(channels,
                                             *[1] * (weight_kernel.dim() - 1))
        kernel[:, :, 0:weight_size[0], 0:weight_size[1]] = weight_kernel

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.
                format(dim))

    def set_weight(self, weight):
        self.weight = weight

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)
class SideWindowGaussianSmoothing(object):
    """ Side window filtering with gaussian kernel.
    Constructure 8 different directions kernel to do filtering. Use the kernel that with the cloest value to original image.
    """
    def __init__(self, iteration, radius, sigma=4):
        half_radius = int(math.ceil(radius / 2))
        self.edge = [
            GaussianSmoothing(1,
                              weight_size=[radius, half_radius],
                              kernel_size=[radius, radius],
                              sigma=sigma,
                              dim=2) for _ in range(4)
        ]
        self.corner = [
            GaussianSmoothing(1,
                              weight_size=[half_radius, half_radius],
                              kernel_size=[radius, radius],
                              sigma=sigma,
                              dim=2) for _ in range(4)
        ]
        self.iteration = iteration

        for k in range(1, 4):
            self.edge[k].set_weight(
                torch.rot90(self.edge[0].weight, k=k, dims=(2, 3)))
            self.corner[k].set_weight(
                torch.rot90(self.corner[0].weight, k=k, dims=(2, 3)))

    def __call__(self, x):
        n, c, h, w = x.shape
        diff = torch.zeros(n, 8, h, w, dtype=torch.float)
        y = x.clone()

        for ch in range(c):
            img = x[:, ch, ::].clone().view(n, 1, h, w)
            for _ in range(self.iteration):
                for k in range(4):
                    diff[:, k, ::] = F.interpolate(self.edge[k](img),
                                                   (h, w)) - img
                    diff[:, k + 4, ::] = F.interpolate(self.corner[k](img),
                                                       (h, w)) - img
                abs_diff = torch.abs(diff)
                mask = torch.argmin(abs_diff, dim=1, keepdim=True)
                masked_diff = torch.gather(input=diff, dim=1, index=mask)
                img += masked_diff
            y[:, ch, ::] = img
        return y

class RecurrentModel(nn.Module):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = opts.n_recurrent
        self.upscale = opts.upscale


        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G_I = get_generator_guide(opts)
        self.networks.append(self.net_G_I)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            param = list(self.net_G_I.parameters()) #+ list(self.net_G_K.parameters())
            self.optimizer_G = torch.optim.Adam(param,
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.opts = opts

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

        dcs_K = []
        for i in range(self.n_recurrent):
            dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.ref_image_full = data['ref_image_full'].to(self.device)
        self.ref_image_sub = data['ref_image_sub'].to(self.device)
        self.tag_kspace_full = data['tag_kspace_full'].to(self.device)
        self.tag_image_full = data['tag_image_full'].to(self.device)
        self.tag_image_sub = data['tag_image_sub'].to(self.device)
        self.tag_kspace_mask2d = data['tag_kspace_mask2d'].to(self.device)

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        I = self.tag_image_sub
        I.requires_grad_(True)
        # bs, _, _, _ = I.size()
        T1 = self.ref_image_full
        T1.requires_grad_(True)


        net = {}
        for i in range(1, self.n_recurrent + 1):
            '''Image Space'''
            # net['r%d_img_pred' % i] = self.net_G_I(I)
            x_I_lr = I
            x_I_t1 = T1

            net['r%d_img_pred' % i] = self.net_G_I(x_I_lr,x_I_t1)  # output recon image [b,c,w,h]
            net['r%d_img_dc_pred' % i], _ = self.dcs_I[i - 1](net['r%d_img_pred' % i], self.tag_kspace_full, self.tag_kspace_mask2d)
            net['r%d_kspc_img_dc_pred' % i] = fft2_net(net['r%d_img_dc_pred' % i])

        self.net = net
        self.recon = net['r%d_img_dc_pred' % i]

    def update_G(self):
        loss_G_L1 = 0
        self.optimizer_G.zero_grad()


        loss_img_l1 = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_l1 = loss_img_l1 + self.criterion(self.net['r%d_img_pred' % j], self.tag_image_full)
        loss_kspc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_kspc = loss_kspc + self.mse(self.net['r%d_kspc_img_dc_pred' % j], self.tag_kspace_full) * 0.0001

        loss_G_L1 = loss_img_l1 + loss_kspc
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img_l1 = loss_img_l1.item()
        self.loss_kspc = loss_kspc.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            message += 'G_L1: {:.4f} Img_L1: {:.4f} dc_loss: {:.4f}'.format(self.loss_G_L1, self.loss_img_l1, self.loss_kspc)
        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G_I.module.load_state_dict(checkpoint['net_G_I'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            if self.opts.wr_L1 > 0:
                psnr_recon = psnr(complex_abs_eval(self.recon),
                                  complex_abs_eval(self.tag_image_full))
                avg_psnr.update(psnr_recon)

                ssim_recon = ssim(complex_abs_eval(self.recon)[0,0,:,:].cpu().numpy(),
                                  complex_abs_eval(self.tag_image_full)[0,0,:,:].cpu().numpy())
                avg_ssim.update(ssim_recon)

                recon_images.append(self.recon[0].cpu())
                gt_images.append(self.tag_image_full[0].cpu())
                input_images.append(self.tag_image_sub[0].cpu())

            message = 'PSNR: {:4f} '.format(avg_psnr.avg)
            message += 'SSIM: {:4f} '.format(avg_ssim.avg)
            val_bar.set_description(desc=message)

        self.psnr_recon = avg_psnr.avg
        self.ssim_recon = avg_ssim.avg

        self.results = {}
        if self.opts.wr_L1 > 0:
            self.results['recon'] = torch.stack(recon_images).squeeze().numpy()
            self.results['gt'] = torch.stack(gt_images).squeeze().numpy()
            self.results['input'] = torch.stack(input_images).squeeze().numpy()