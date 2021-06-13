import os
import sys
from collections import OrderedDict
from absl import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from .base import AutoEncoderModule
from elm.nn import ConvNN, ConvTransposeNN, MLP

class ConvAE(AutoEncoderModule):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               n_channels,
               img_size, 
               dim_latent, 
               activation=F.relu,
               readout_fn=None, 
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
    
    self._create_networks()
    self.params = self.parameters()
    
    logging.debug("---------       Conv AE       ---------")
    logging.debug("-------- Trainable Variables ---------")
    for name, p in self.named_parameters():
      logging.debug("{}, {}".format(name, p.size()))
    logging.debug("--------------------------------------")
    
  def _create_networks(self):
    nc= self.n_channels
    self.encoder = torch.nn.Sequential(OrderedDict([
      ("conv_nn", ConvNN(in_channels=self.in_channels, 
                        out_channels=[nc,nc,2*nc,2*nc,2*nc], 
                        kernel_size=[3,3,3,3,5],
                        stride=[2,2,2,2,2],
                        padding_mode='constant',
                        activation=self.activation,
                        out_activation=self.activation,
                        use_bias=True)),
      ("flatten", torch.nn.Flatten()),
      ("mlp", MLP(in_sizes=2*2*2*nc, 
                  out_sizes=[self.dim_latent]))
    ]))
    self.decoder = torch.nn.Sequential(OrderedDict([
      ("mlp", MLP(in_sizes=self.dim_latent, 
          out_sizes=[2*nc*2*2], 
          out_activation=self.activation)),
      ("unflatten", torch.nn.Unflatten(dim=1, unflattened_size=(2*nc,2,2))),
      ("conv_transpose_nn", ConvTransposeNN(in_channels=2*nc, 
             out_channels=[2*nc,2*nc,nc,nc,nc,self.out_channels], 
             kernel_size=[5,3,3,3,3,3],
             stride=[2,2,2,2,2,1],
             padding_mode='constant',
             activation=self.activation,
             out_activation=self.readout_fn,
             use_bias=True))
    ]))
    
    
  def encode(self, x):
    return self.encoder(x)
  
  def decode(self, z):
    return self.decoder(z)
    
  def reconstruct(self, x):
    z = self.encode(x)
    x_hat = self.decode(z)
    return x_hat 
  
  def forward(self, x):
    z = self.encode(x)
    return z 
