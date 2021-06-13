import os
import sys
from collections import OrderedDict
from absl import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import torch.optim as optim
import pytorch_lightning as pl
from .base import AutoEncoderModule
from elm.nn import EquivariantConvNN, EquivariantConvTransposeNN, MLP, EquivariantGConvNN, EquivariantGConvTransposeNN
from elm.utils import get_meshgrid, CyclicGArray, DihedralGArray, Rot2dOnCyclicGArray, FlipRot2dOnCyclicGArray, recursive_decomposition, product_of_representives
from elm.utils import visualize_2d_vector

class EquivariantAE(AutoEncoderModule):
  
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
               use_g_offset=False,
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
    self.avg_pool_size = avg_pool_size
    self.use_g_offset = True
    self._create_networks()
    self.params = self.parameters()

    logging.debug("---------       Equivariant AE       ---------")
    logging.debug("-------- Trainable Variables ---------")
    for name, p in self.named_parameters():
      logging.debug("{}, {}".format(name, p.size()))
    logging.debug("--------------------------------------")
    
  def _create_networks(self):
    nc = self.n_channels
    if self.fiber_group == 'trivial':
      self.quotient_orders = np.array([(2,), (2,), (2,), (2,), (self.img_size//(2**4),)])
      self.encoder = EquivariantConvNN(in_channels=self.in_channels, 
                          out_channels=[nc,nc,2*nc,2*nc,2*nc], 
                          kernel_size=[3,3,3,3,5],
                          scale_factor=self.quotient_orders,
                          padding_mode='circular',
                          activation=self.activation,
                          out_activation=self.activation,
                          use_bias=True)
      self.decoder = EquivariantConvTransposeNN(in_channels=2*nc, 
              out_channels=[2*nc,2*nc,nc,nc,nc,self.out_channels], 
              kernel_size=[5,3,3,3,3,3],
              scale_factor=np.concatenate([self.quotient_orders[::-1], np.ones((1,1), dtype=np.int32)]),
              padding_mode='circular',
              activation=self.activation,
              out_activation=self.readout_fn,
              use_bias=True,
              avg_pool_size=self.avg_pool_size)
    elif self.fiber_group == 'rot_2d':
      self.quotient_orders = np.array([(2,1), (2,1), (2,1), (2,2), (self.img_size//(2**4),2)])
      self.encoder = EquivariantGConvNN(in_channels=self.in_channels, 
                          out_channels=[nc,nc,2*nc,2*nc,2*nc], 
                          kernel_size=[3,3,3,3,5],
                          scale_factor=self.quotient_orders,
                          padding_mode='circular',
                          activation=self.activation,
                          out_activation=self.activation,
                          use_bias=True,
                          fiber_group=self.fiber_group,
                          n_rot=self.n_rot)
      self.decoder = EquivariantGConvTransposeNN(in_channels=2*nc, 
              out_channels=[2*nc,2*nc,nc,nc,nc,self.out_channels], 
              kernel_size=[5,3,3,3,3,3],
              scale_factor=np.concatenate([self.quotient_orders[::-1], np.ones((1,2), dtype=np.int32)]),
              padding_mode='circular',
              activation=self.activation,
              out_activation=self.readout_fn,
              use_bias=True,
              fiber_group=self.fiber_group,
              n_rot=self.n_rot,
              avg_pool_size=self.avg_pool_size)
    elif self.fiber_group == 'flip_rot_2d':
      self.quotient_orders = np.array([(2,1,1), (2,1,1), (2,1,1), (2,2,1), (self.img_size//(2**4),2,2)])
      self.encoder = EquivariantGConvNN(in_channels=self.in_channels, 
                          out_channels=[nc,nc,2*nc,2*nc,2*nc], 
                          kernel_size=[3,3,3,3,5],
                          scale_factor=self.quotient_orders,
                          padding_mode='circular',
                          activation=self.activation,
                          out_activation=self.activation,
                          use_bias=True,
                          fiber_group=self.fiber_group,
                          n_rot=self.n_rot)
      self.decoder = EquivariantGConvTransposeNN(in_channels=2*nc, 
              out_channels=[2*nc,2*nc,nc,nc,nc,self.out_channels], 
              kernel_size=[5,3,3,3,3,3],
              scale_factor=np.concatenate([self.quotient_orders[::-1], np.ones((1,3), dtype=np.int32)]),
              padding_mode='circular',
              activation=self.activation,
              out_activation=self.readout_fn,
              use_bias=True,
              fiber_group=self.fiber_group,
              n_rot=self.n_rot,
              avg_pool_size=self.avg_pool_size)
    else:
      raise Exception("Unknown fiber group {}".format(self.fiber_group))
    
    self.flatten = torch.nn.Flatten()
    self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(2*nc,1,1))
    self.encoder_mlp = MLP(in_sizes=2*nc, out_sizes=[self.dim_latent])
    self.decoder_mlp = MLP(in_sizes=self.dim_latent, out_sizes=[2*nc], out_activation=self.activation)
  
  def g_act_z(self, g, z_eqv, origin=(0, 0)):
    b = z_eqv.elems.size()[0]
    g = torch.Tensor([g]).repeat(b,1).type_as(z_eqv.elems)
    T = CyclicGArray(elems=torch.Tensor([origin]).repeat(b,1).type_as(z_eqv.elems), N=self.img_size, D=2)
    if not self.fiber_group == 'trivial':
      z_eqv.garray_N = T.inv().mul(z_eqv.garray_N)
    if self.fiber_group == 'trivial':
      G = CyclicGArray(elems=g[:, :2], N=self.img_size, D=2)
    elif self.fiber_group == 'rot_2d':
      F = CyclicGArray(elems=g[:, 2:], N=self.n_rot)
      B = CyclicGArray(elems=g[:, :2], N=self.img_size, D=2)
      G = Rot2dOnCyclicGArray(B, F)
    elif self.fiber_group == 'flip_rot_2d':
      F = DihedralGArray(elems=g[:, 2:], N=self.n_rot)
      B = CyclicGArray(elems=g[:, :2], N=self.img_size, D=2)
      G = FlipRot2dOnCyclicGArray(B, F)
    else:
      raise Exception("Unknown fiber group {}".format(self.fiber_group))
    z_eqv = G.mul(z_eqv)
    if not self.fiber_group == 'trivial':
      z_eqv.garray_N = T.mul(z_eqv.garray_N)
    crs = recursive_decomposition(z_eqv, quotient_orders=self.quotient_orders, fiber_group=self.fiber_group)
    return crs, z_eqv
  
  def g_act_x(self, g, x, origin=(0, 0), n_rot=4):
    if isinstance(g, np.ndarray):
      g = g.tolist()
    x = torch.roll(x, shifts=[32-origin[0],32-origin[1]], dims=[-2,-1])
    if len(g) >= 4 and g[3] > 0:
      x = TF.vflip(x)
      x = torch.roll(x, shifts=-self.img_size+1, dims=-2)
    if len(g) >= 3 and g[2] > 0:
      angle = (360/n_rot) * g[2]
      rad = angle / 180 * np.pi
      d = 0.5*np.sqrt(2) 
      r1 = np.array([np.cos(np.pi/4) * d, np.sin(np.pi/4) * d])
      r2 = np.array([np.cos(rad+np.pi/4) * d, np.sin(rad+np.pi/4) * d])
      r = tuple((r2 - r1).round().astype(np.int32).tolist())
      x = TF.rotate(x, angle)
      x = torch.roll(x, shifts=r, dims=[-2,-1])
    if len(g) >= 2 and np.abs(g[:2]).max() > 0:
      x = torch.roll(x, shifts=[g[0], g[1]], dims=[-2,-1])
    x = torch.roll(x, shifts=[origin[0]-32,origin[1]-32], dims=[-2,-1])
    return x

  def encode(self, x):
    
    z_inv, crs, z_eqv = self.encoder(x)
    z_inv = self.encoder_mlp(self.flatten(z_inv))
    return z_inv, crs, z_eqv
  
  def decode(self, z_inv, crs):
    z_inv = self.unflatten(self.decoder_mlp(z_inv))
    x_hat = self.decoder(z_inv, crs)
    return x_hat 
    
  def reconstruct(self, x, g=None, origin=(0,0)):

    z_inv, crs, z_eqv = self.encode(x)
  
    if g is not None:
      crs, z_eqv = self.g_act_z(g, z_eqv, origin)
    x_hat = self.decode(z_inv, crs)
    
    return x_hat
  
  def forward(self, x):
    z_inv, _, z_eqv = self.encode(x)
    return torch.cat([z_inv, z_eqv.elems], dim=1)
  
    
  
  
  