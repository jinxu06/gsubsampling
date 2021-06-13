from collections import deque
import numpy as np
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
import e2cnn.gspaces
import e2cnn.nn
from .conv_nn import BaseConvNN, ConvNN
from .subsampling import EquivariantSubSampling, EquivariantFeatureSpace2Group
from .upsampling import EquivariantUpSampling
from elm.utils import CyclicGArray, DihedralGArray, Rot2dOnCyclicGArray, FlipRot2dOnCyclicGArray, recursive_decomposition, product_of_representives
from elm.utils import get_same_pad, calculate_nonlinearity_gain

def gspace(n_rot=1, n_flip=1):
  n_rot, n_flip = int(n_rot), int(n_flip)
  if n_rot > 1 and n_flip > 1:
    return e2cnn.gspaces.FlipRot2dOnR2(N=n_rot, axis=0.)
  elif n_rot > 1 and n_flip == 1:
    return e2cnn.gspaces.Rot2dOnR2(N=n_rot)
  elif n_rot == 1 and n_flip > 1:
    return e2cnn.gspaces.Flip2dOnR2(axis=0.)
  elif n_rot == 1 and n_flip == 1:
    return e2cnn.gspaces.TrivialOnR2()
  
class EquivariantGConvNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               scale_factor=1,
               padding_mode='circular',
               activation=F.relu,
               out_activation=None,
               use_bias=True,
               fiber_group='rot_2d',
               n_rot=4,
               before_activation=False,
               device='cuda'):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias)
    self.scale_factor = np.array(scale_factor)
    self.before_activation = before_activation
    self.fiber_group = fiber_group 
    self.n_rot = n_rot
    self.n_flip = 2 if self.fiber_group == 'flip_rot_2d' else 1
    self.gspaces = []
    if self.fiber_group == 'rot_2d':
      fiber_group_sizes = np.array([[self.n_rot,]], dtype=np.int32)
    elif self.fiber_group == 'flip_rot_2d':
      fiber_group_sizes = np.array([[self.n_rot, self.n_flip]], dtype=np.int32)
    for i in range(self.num_layers):
      self.gspaces.append(gspace(*tuple(fiber_group_sizes[-1])))
      fiber_group_sizes = np.append(fiber_group_sizes, [(fiber_group_sizes[-1] / self.scale_factor[i, 1:]).astype(np.int32)], axis=0)
    self.out_tensor_types = [e2cnn.nn.FieldType(self.gspaces[i], \
                            self.out_channels[i]*[self.gspaces[i].regular_repr]) \
                            for i in range(self.num_layers)]
    self.in_tensor_types = [e2cnn.nn.FieldType(self.gspaces[0], \
                            self.in_channels[0]*[self.gspaces[0].trivial_repr])] \
                            + [e2cnn.nn.FieldType(self.gspaces[i], \
                            self.in_channels[i]*[self.gspaces[i].regular_repr]) \
                            for i in range(1, self.num_layers)]
    self.f2q = EquivariantFeatureSpace2Group(in_channels=self.out_channels[0], fiber_group=self.fiber_group, fiber_group_size=(self.n_rot, self.n_flip), temperature=0.0001)
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      trivial_repr = self.out_tensor_types[i].representations[0].is_trivial()
      layer['gconv'] = e2cnn.nn.R2Conv(self.in_tensor_types[i], 
                             self.out_tensor_types[i], 
                             kernel_size=self.kernel_size[i], 
                             bias=(trivial_repr and self.use_bias))
      self.add_module("gconv_{}".format(i+1), layer['gconv'])
      layer['subsampling'] = EquivariantSubSampling(scale_factor=self.scale_factor[i], fiber_group_size=fiber_group_sizes[i], fiber_group=self.fiber_group, n_rot=self.n_rot)
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      self.layers.append(layer)
      

  def forward(self, x):
    crs = None
    y = x
    for i, layer in enumerate(self.layers):
      _, _, h, w = y.size()
      pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=1)
      y = F.pad(y, pad, mode=self.padding_mode)
      y = e2cnn.nn.GeometricTensor(y, self.in_tensor_types[i])
      y = layer['gconv'](y)
      y = y.tensor

      if layer['activation'] is not None:
        y = layer['activation'](y)

      if not self.before_activation:
        b, c, h, w = y.size()
        if crs is None:
          z = self.f2q(y)
          if self.fiber_group == "rot_2d":
            garray = Rot2dOnCyclicGArray(garray_N=CyclicGArray(elems=z[:, 2:], N=h, D=2),
                              garray_H=CyclicGArray(elems=z[:, 1:2], N=self.n_rot))
          elif self.fiber_group == "flip_rot_2d":
            garray = FlipRot2dOnCyclicGArray(garray_N=CyclicGArray(elems=z[:, 2:], N=h, D=2),
                              garray_H=DihedralGArray(elems=torch.flip(z[:, :2], dims=(1,)), N=self.n_rot))
          crs = recursive_decomposition(garray, 
                                  quotient_orders=self.scale_factor, 
                                  fiber_group=self.fiber_group)
        y = layer['subsampling'](y, crs[:, i])
    return y, crs, garray
  
  
class EquivariantGConvTransposeNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               scale_factor=1,
               padding_mode='circular',
               activation=F.relu,
               out_activation=None,
               use_bias=True,
               fiber_group='rot_2d',
               n_rot=4,
               avg_pool_size=1):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias)
    
    self.scale_factor = np.array(scale_factor)
    self.fiber_group = fiber_group 
    self.n_rot = n_rot
    self.avg_pool_size = avg_pool_size
    self.n_flip = 2 if self.fiber_group == 'flip_rot_2d' else 1
    
    self.gspaces = []
    if self.fiber_group == 'rot_2d':
      fiber_group_sizes = np.ones((1, 1), dtype=np.int32)
    elif self.fiber_group == 'flip_rot_2d':
      fiber_group_sizes = np.ones((1, 2), dtype=np.int32)
    for i in range(self.num_layers):
      fiber_group_size = (fiber_group_sizes[-1] * self.scale_factor[i, 1:]).astype(np.int32)
      fiber_group_sizes = np.append(fiber_group_sizes, [fiber_group_size], axis=0)
      self.gspaces.append(gspace(*tuple(fiber_group_sizes[-1])))
      
    self.out_tensor_types = [e2cnn.nn.FieldType(self.gspaces[i], \
                            self.out_channels[i]*[self.gspaces[i].regular_repr]) \
                            for i in range(self.num_layers-1)] \
                              + [e2cnn.nn.FieldType(self.gspaces[-1], \
                            self.out_channels[-1]*[self.gspaces[-1].trivial_repr])]
    self.in_tensor_types = [e2cnn.nn.FieldType(self.gspaces[i], \
                            self.in_channels[i]*[self.gspaces[i].regular_repr]) \
                            for i in range(self.num_layers)]
               
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      """
      Transposed convolution can produce artifacts which can harm the overall equivariance of the model.
      We suggest using :class:`~e2cnn.nn.R2Upsampling` combined with :class:`~e2cnn.nn.R2Conv` to perform
      upsampling.
      """
      trivial_repr = self.out_tensor_types[i].representations[0].is_trivial()
      layer['gconv'] = e2cnn.nn.R2Conv(self.in_tensor_types[i], 
                             self.out_tensor_types[i], 
                             kernel_size=self.kernel_size[i], 
                             bias=(trivial_repr and self.use_bias))
      self.add_module("gconv_{}".format(i+1), layer['gconv'])
      layer['upsampling'] = EquivariantUpSampling(scale_factor=self.scale_factor[i], 
                                                    fiber_group_size=fiber_group_sizes[i], 
                                                    fiber_group=self.fiber_group, 
                                                    n_rot=self.n_rot)
      if self.avg_pool_size > 1:
        layer['smoothing'] = torch.nn.AvgPool2d(self.avg_pool_size, stride=1)
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      self.layers.append(layer)
      

  def forward(self, x, crs):
    y = x
    crs = deque(torch.unbind(crs, dim=1))
    for i, layer in enumerate(self.layers):
      if 'upsampling' in layer and (np.mean(self.scale_factor[i])>1):
        y = layer['upsampling'](y, crs.pop())
        y = y * (np.prod(self.scale_factor[i]) * self.scale_factor[i, 0]) ** 0.5
      _, _, h, w = y.size()
      pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=1)
      y = F.pad(y, pad, mode=self.padding_mode)
      y = e2cnn.nn.GeometricTensor(y, self.in_tensor_types[i])
      y = layer['gconv'](y)
      y = y.tensor
      if self.avg_pool_size > 1:
        _, _, h, w = y.size()
        pad = get_same_pad(size=[h, w], kernel_size=self.avg_pool_size, stride=1)
        y = F.pad(y, pad=pad, mode=self.padding_mode)
        y = layer['smoothing'](y)
        
      if layer['activation'] is not None:
        y = layer['activation'](y)
    return y

    
  

  
  