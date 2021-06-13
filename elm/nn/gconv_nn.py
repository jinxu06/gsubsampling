import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
import e2cnn.gspaces
import e2cnn.nn
from elm.utils import get_same_pad, calculate_nonlinearity_gain


class BaseGConvNN(torch.nn.Module): 
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None,
               use_bias=False,
               fiber_group='rot_2d',
               n_rot=4,
               device='cuda'):
    super().__init__()
    assert isinstance(in_channels, int), "in_channels should be an instance of int"
    assert isinstance(out_channels, list), "in_channels should be an instance of list"
    self.in_channels = [in_channels] + out_channels[:-1]
    self.out_channels = out_channels
    self.num_layers = len(out_channels)
    if isinstance(kernel_size, int) or isinstance(kernel_size, tuple):
      kernel_size = [kernel_size for i in range(self.num_layers)]
    else:
      assert len(kernel_size) == self.num_layers, "length of kernel_size should be {0} (num_layers), but is {1}".format(self.num_layers, len(kernel_size))
    self.kernel_size = kernel_size 
    if isinstance(stride, int) or isinstance(stride, tuple):
      stride = [stride for i in range(self.num_layers)]
    else:
      assert len(stride) == self.num_layers, "length of stride should be {0} (num_layers), but is {1}".format(self.num_layers, len(stride))
    self.stride = stride
    self.padding_mode = padding_mode
    self.activation = activation 
    self.out_activation = out_activation
    self.use_bias = use_bias 
    self.fiber_group = fiber_group
    self.n_rot = n_rot 
    self.device = device
    if self.fiber_group == 'rot_2d':
      self.gspace = e2cnn.gspaces.Rot2dOnR2(N=self.n_rot)
    elif self.fiber_group == 'flip_rot_2d':
      self.gspace = e2cnn.gspaces.FlipRot2dOnR2(N=self.n_rot)
    else:
      raise Exception("fiber_group {} is not supported".format(self.fiber_group))

class GConvNN(BaseGConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None,
               use_bias=False,
               fiber_group='rot_2d',
               n_rot=4,
               device='cuda'):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias,
                     fiber_group=fiber_group,
                     n_rot=n_rot,
                     device=device)
    
    self.out_tensor_types = [e2cnn.nn.FieldType(self.gspace, \
                            self.out_channels[i]*[self.gspace.regular_repr]) \
                            for i in range(self.num_layers)]
    self.in_tensor_types = [e2cnn.nn.FieldType(self.gspace, \
                            self.in_channels[0]*[self.gspace.trivial_repr])] \
                            + self.out_tensor_types[:-1]
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      trivial_repr = self.out_tensor_types[i].representations[0].is_trivial()
      layer['gconv'] = e2cnn.nn.R2Conv(self.in_tensor_types[i], 
                             self.out_tensor_types[i], 
                             kernel_size=self.kernel_size[i], 
                             stride=self.stride[i],
                             bias=(trivial_repr and self.use_bias))
      self.add_module("gconv_{}".format(i+1), layer['gconv'])
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      self.layers.append(layer)
      

  def forward(self, x):
    y = x
    for i, layer in enumerate(self.layers):
      _, _, h, w = y.size()
      pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=self.stride[i])
      y = F.pad(y, pad, mode=self.padding_mode)
      y = e2cnn.nn.GeometricTensor(y, self.in_tensor_types[i])
      y = layer['gconv'](y)
      y = y.tensor
      if layer['activation'] is not None:
        y = layer['activation'](y)
    return y


class GConvTransposeNN(BaseGConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None,
               use_bias=False,
               fiber_group='rot_2d',
               n_rot=4,
               device='cuda'):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias,
                     fiber_group=fiber_group,
                     n_rot=n_rot,
                     device=device)
               
    self.in_tensor_types = [e2cnn.nn.FieldType(self.gspace, \
                            self.in_channels[i]*[self.gspace.regular_repr]) \
                            for i in range(self.num_layers)]
    self.out_tensor_types = self.in_tensor_types[1:] + \
                            [e2cnn.nn.FieldType(self.gspace, \
                            self.out_channels[-1]*[self.gspace.trivial_repr])]                   
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
      if self.stride[i] > 1:
        layer['upsampling'] = e2cnn.nn.R2Upsampling(self.in_tensor_types[i], self.stride[i], mode='bilinear')
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      self.layers.append(layer)
      

  def forward(self, x):
    y = x
    for i, layer in enumerate(self.layers):
      if 'upsampling' in layer:
        y = e2cnn.nn.GeometricTensor(y, self.in_tensor_types[i])
        y = layer['upsampling'](y)
        y = y.tensor
      _, _, h, w = y.size()
      pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=1)
      y = F.pad(y, pad, mode=self.padding_mode)
      y = e2cnn.nn.GeometricTensor(y, self.in_tensor_types[i])
      y = layer['gconv'](y)
      y = y.tensor
      if layer['activation'] is not None:
        y = layer['activation'](y)
    return y
