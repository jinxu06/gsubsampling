import os
import sys
from collections import OrderedDict
from absl import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as tdist 
import pytorch_lightning as pl
from .base import AutoEncoderModule, VariationalAutoEncoderModule
from elm.nn import ConvNN, ConvTransposeNN, MLP

class ConvVAE(VariationalAutoEncoderModule):
  
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
  
    logging.debug("---------       Conv VAE       ---------")
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
                  out_sizes=[2*self.dim_latent]))
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
    mu, log_sigma_sq = torch.chunk(self.encoder(x), chunks=2, dim=-1)
    return mu, log_sigma_sq
  
  def decode(self, z):
    x_hat = self.decoder(z)
    return x_hat 
  
  def forward(self, x):
    mu, _ = self.encode(x)
    return mu
  
  def reparameterize(self, mu, log_sigma_sq):
    sigma = torch.exp(log_sigma_sq/2.)
    eps = torch.normal(torch.zeros_like(mu), torch.ones_like(sigma))
    return eps * sigma + mu
  
  def reconstruct(self, x):
    mu, _ = self.encode(x)
    x_hat = self.decode(mu)
    return x_hat 
  
  def generate(self, n_samples=16):
    z = torch.normal(torch.zeros(n_samples, self.dim_latent), torch.ones(n_samples, self.dim_latent))
    x_hat = self.decode(z)
    return x_hat
    
  def compute_loss_and_metrics(self, x, y=None):
    mu, log_sigma_sq = self.encode(x)
    z = self.reparameterize(mu, log_sigma_sq)
    x_hat = self.decode(z)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size()[0]
    kl_loss = torch.sum(-0.5 * torch.sum(1 + log_sigma_sq - mu ** 2 - log_sigma_sq.exp(), dim = 1), dim = 0) / x.size()[0]
    loss = recon_loss + kl_loss
    logs = {
        "recon": recon_loss,
        "kl": kl_loss,
        "elbo": loss
    }
    return loss, logs 
    
  
  

  

  
