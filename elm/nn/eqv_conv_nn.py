from collections import deque
import numpy as np
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from .conv_nn import BaseConvNN, ConvNN
from .subsampling import EquivariantSubSampling, EquivariantFeatureSpace2Group
from .upsampling import EquivariantUpSampling
from elm.utils import get_same_pad, kaiming_init
from elm.utils import CyclicGArray, Rot2dOnCyclicGArray, FlipRot2dOnCyclicGArray, recursive_decomposition, product_of_representives

class EquivariantConvNN(ConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               scale_factor=1,
               padding_mode='circular',
               activation=F.relu,
               out_activation=None,
               use_bias=True,
               before_activation=False,
               device='cuda'):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias,
                     device=device)
    
    self.scale_factor = scale_factor
    self.before_activation = before_activation
    self.f2q = EquivariantFeatureSpace2Group(in_channels=self.out_channels[0],
                                             fiber_group='trivial', 
                                             temperature=0.0001)
    for i, layer in enumerate(self.layers):
      layer['subsampling'] = EquivariantSubSampling(scale_factor=self.scale_factor[i, 0], fiber_group='trivial')
      
  def forward(self, x):
    _, _, h, w = x.size()
    y = x  
    crs = None
    for i, layer in enumerate(self.layers):
      _, _, h, w = y.size()
      pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=1)
      y = F.pad(y, pad, mode=self.padding_mode)
      y = layer['conv'](y)
      if layer['activation'] is not None:
        y = layer['activation'](y)
      if not self.before_activation and self.scale_factor[i, 0] > 1:
        if crs is None:
          z = self.f2q(y)
          garray = CyclicGArray(elems=z[:, 2:], N=h, D=2)
          crs = recursive_decomposition(garray, quotient_orders=self.scale_factor, fiber_group='trivial')
        y = layer['subsampling'](y, crs[:, i])
    return y, crs, garray
  
  
class EquivariantConvTransposeNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               scale_factor=1,
               padding_mode='circular',
               activation=F.relu,
               out_activation=None,
               use_bias=True,
               avg_pool_size=1,
               device='cuda'):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias,
                     device=device)
    self.scale_factor = scale_factor
    self.avg_pool_size = avg_pool_size
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      layer['upsampling'] = EquivariantUpSampling(scale_factor=self.scale_factor[i, 0], fiber_group='trivial')
      layer['conv_transpose'] = torch.nn.ConvTranspose2d(self.in_channels[i],
                                      self.out_channels[i], 
                                      kernel_size=self.kernel_size[i], 
                                      stride=1, 
                                      padding=self.kernel_size[i]-1, 
                                      bias=self.use_bias)
      self.add_module("conv_transpose_{}".format(i+1), layer['conv_transpose'])
      if self.avg_pool_size > 1:
        layer['smoothing'] = torch.nn.AvgPool2d(self.avg_pool_size, stride=1)
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      kaiming_init(layer['conv_transpose'], F.relu, 1, mode='fan_out', use_bias=self.use_bias)
      self.layers.append(layer)
        
      
  def forward(self, x, crs):
    y = x  
    crs = deque(torch.unbind(crs, dim=1))
    for i, layer in enumerate(self.layers):
      if self.scale_factor[i, 0] > 1:
        y = layer['upsampling'](y, crs.pop())
        y = y * self.scale_factor[i, 0]
      y = F.pad(y, pad=[(self.kernel_size[i]-1)//2 for _ in range(4)], mode=self.padding_mode)
      y = layer['conv_transpose'](y)
      if self.avg_pool_size > 1:
        _, _, h, w = y.size()
        pad = get_same_pad(size=[h, w], kernel_size=self.avg_pool_size, stride=1)
        y = F.pad(y, pad=pad, mode=self.padding_mode)
        y = layer['smoothing'](y)
      if layer['activation'] is not None:
        y = layer['activation'](y)
    return y

    
  

  
  