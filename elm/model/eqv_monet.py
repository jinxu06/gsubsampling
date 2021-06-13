from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

import numpy as np

from genesis.modules.unet import UNet
import genesis.modules.seq_att as seq_att
from genesis.utils.misc import get_kl
from .base import AutoEncoderModule
from absl import logging
from genesis.utils.misc import average_ari

from .monet import MONet
from .eqv_vae import EquivariantVAE

class EquivariantComponentVAE(EquivariantVAE):
  
  def __init__(self, 
                in_channels, 
                out_channels,
                n_channels,
                img_size, 
                dim_latent, 
                activation=F.relu,
                readout_fn=None, 
                fiber_group='trivial',
                n_rot=1,
                avg_pool_size=1,
                optim_lr=0.0001, 
                profiler=None):
    super().__init__(in_channels=in_channels, 
                    out_channels=out_channels+1,
                    n_channels=n_channels,
                    img_size=img_size, 
                    dim_latent=dim_latent, 
                    activation=activation,
                    readout_fn=readout_fn, 
                    fiber_group=fiber_group,
                    n_rot=n_rot,
                    avg_pool_size=avg_pool_size,
                    optim_lr=optim_lr, 
                    profiler=profiler)

    
  def forward(self, x, log_mask):
    
    K = 1
    b_sz = x.size(0)
    if isinstance(log_mask, list) or isinstance(log_mask, tuple):
      K = len(log_mask)
      # Repeat x along batch dimension
      x = x.repeat(K, 1, 1, 1)
      # Concat log_m_k along batch dimension
      log_mask = torch.cat(log_mask, dim=0)

    # -- Encode
    mask = log_mask.exp()
    x *= mask 
    x = torch.cat((x, mask), dim=1)
    
    mu, log_sigma_sq, crs, z_eqv = self.encode(x)
    sigma = torch.exp(log_sigma_sq / 2.)
    z = self.reparameterize(mu, log_sigma_sq)
    x_r = self.decode(z, crs)

    # -- Track quantities of interest and return
    x_r_k = torch.chunk(x_r, K, dim=0)
    z_k = torch.chunk(z, K, dim=0)
    mu_k = torch.chunk(mu, K, dim=0)
    sigma_k = torch.chunk(sigma, K, dim=0)
    stats = AttrDict(mu_k=mu_k, sigma_k=sigma_k, z_k=z_k)
    return x_r_k, stats
  

class EquivariantMONet(MONet):
  
  def __init__(self, 
                in_channels,
                out_channels,
                n_channels,
                img_size,
                dim_latent,
                activation=torch.nn.ReLU(),
                K_steps=5,
                prior_mode='softmax',
                montecarlo_kl=False,
                pixel_bound=True,
                kl_l_beta=0.5,
                kl_m_beta=0.5,
                pixel_std_fg=0.1,
                pixel_std_bg=0.1,
                optimizer='ADAM',
                fiber_group='trivial',
                n_rot=1,
                avg_pool_size=3):
    self.ldim = dim_latent
    self.fiber_group = fiber_group
    self.n_rot = n_rot
    self.avg_pool_size = avg_pool_size
    super().__init__(in_channels=in_channels,
                      out_channels=out_channels,
                      n_channels=n_channels,
                      img_size=img_size,
                      dim_latent=dim_latent,
                      activation=activation,
                      K_steps=K_steps,
                      prior_mode=prior_mode,
                      montecarlo_kl=montecarlo_kl,
                      pixel_bound=pixel_bound,
                      kl_l_beta=kl_l_beta,
                      kl_m_beta=kl_m_beta,
                      pixel_std_fg=pixel_std_fg,
                      pixel_std_bg=pixel_std_bg,
                      optimizer=optimizer)
    
    
  def _create_networks(self):
    
    core = UNet(int(np.log2(self.img_size)-1), 32)
    self.att_process = seq_att.SimpleSBP(core)
    # - Component VAE
    self.comp_vae = EquivariantComponentVAE(self.in_channels+1, 
                                  self.out_channels,
                                  self.n_channels,
                                  self.img_size, 
                                  self.dim_latent, 
                                  activation=F.relu,
                                  readout_fn=None, 
                                  fiber_group=self.fiber_group,
                                  n_rot=self.n_rot,
                                  avg_pool_size=self.avg_pool_size)
    self.comp_vae.pixel_bound = False
    
    
                      