import os
import sys
from collections import OrderedDict
from absl import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import e2cnn.gspaces
import e2cnn.nn
from .base import AutoEncoderModule
from elm.nn import MLP, GConvNN, GConvTransposeNN


class GConvAE(AutoEncoderModule):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               n_channels,
               img_size, 
               dim_latent, 
               activation=F.relu,
               readout_fn=None, 
               fiber_group='rot_2d',
               n_rot=4,
               optim_lr=0.0001, 
               profiler=None):
    
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     n_channels=n_channels,
                     img_size=img_size,
                     dim_latent=dim_latent, 
                     activation=activation,
                     readout_fn=readout_fn,
                     optim_lr=optim_lr, 
                     profiler=profiler)
    self.fiber_group = fiber_group
    self.n_rot = n_rot 
    self._create_networks()
    self.params = self.parameters()

    logging.debug("--------       GConv AE       ---------")
    logging.debug("-------- Trainable Variables ---------")
    for name, p in self.named_parameters():
      logging.debug("{}, {}".format(name, p.size()))
    logging.debug("--------------------------------------")
    
  def _create_networks(self):
    self.n_flip = 1
    if 'flip' in self.fiber_group:
      self.n_flip = 2
    nc = self.n_channels
    self.encoder = torch.nn.Sequential(
                    GConvNN(in_channels=self.in_channels, 
                    out_channels=[nc,nc,2*nc,2*nc,2*nc],
                    kernel_size=[3,3,3,3,5],
                    stride=[2,2,2,2,2],
                    padding_mode='circular',
                    activation=self.activation,
                    out_activation=None,
                    use_bias=True,
                    fiber_group=self.fiber_group,
                    n_rot=self.n_rot))
    self.flatten = torch.nn.Flatten()
    self.encoder_mlp = MLP(in_sizes=self.n_rot*self.n_flip*2*2*2*nc, 
                  out_sizes=[self.dim_latent])
    self.decoder = torch.nn.Sequential(
                    GConvTransposeNN(in_channels=2*nc, 
                    out_channels=[2*nc,2*nc,nc,nc,nc,self.in_channels],  
                    kernel_size=[5,3,3,3,3,3],
                    stride=[2,2,2,2,2,1],
                    padding_mode='circular',
                    activation=self.activation,
                    out_activation=self.readout_fn,
                    use_bias=True,
                    fiber_group=self.fiber_group,
                    n_rot=self.n_rot))
    self.decoder_mlp = MLP(in_sizes=self.dim_latent, 
          out_sizes=[self.n_rot*self.n_flip*2*2*2*nc], 
          out_activation=self.activation)
    self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(2*nc*self.n_flip*self.n_rot,2,2))
    
  def encode(self, x):
    z = self.encoder(x)
    z = self.encoder_mlp(self.flatten(z))
    return z
  
  def decode(self, z):
    z = self.unflatten(self.decoder_mlp(z))
    x_hat = self.decoder(z)
    return x_hat 
    
  def reconstruct(self, x):
    
    z = self.encode(x)
    x_hat = self.decode(z)
    
    return x_hat 
  
  def forward(self, x):
    z = self.encode(x)
    return z 