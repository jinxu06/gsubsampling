import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from elm.utils import get_same_pad, calculate_nonlinearity_gain, kaiming_init

class BaseConvNN(torch.nn.Module):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None, 
               use_bias=True,
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
    self.device = device

class ConvNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None, 
               use_bias=True,
               device='cuda'):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias,
                     device=device)
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      layer['conv'] = torch.nn.Conv2d(self.in_channels[i],
                                      self.out_channels[i], 
                                      kernel_size=self.kernel_size[i], 
                                      stride=self.stride[i], 
                                      bias=self.use_bias)
      self.add_module("conv_{}".format(i+1), layer['conv'])
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      kaiming_init(layer['conv'], F.relu, 1, mode='fan_in', use_bias=self.use_bias)
      self.layers.append(layer)
      
  def forward(self, x):
    y = x
    for i, layer in enumerate(self.layers):
      _, _, h, w = y.size()
      pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=self.stride[i])
      y = F.pad(y, pad, mode=self.padding_mode)
      y = layer['conv'](y)
      if layer['activation'] is not None:
        y = layer['activation'](y)
    return y
  
class ConvTransposeNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None, 
               use_bias=True,
               device='cuda'):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias,
                     device=device)
    
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      if self.kernel_size[i] % 2 == 1:
        padding = (self.kernel_size[i]-1) // 2
        output_padding = self.stride[i] - 1
      else:
        raise Exception("Currently only odd kernel size is properly handled")
      layer['conv_transpose'] = torch.nn.ConvTranspose2d(self.in_channels[i],
                                      self.out_channels[i], 
                                      kernel_size=self.kernel_size[i], 
                                      stride=self.stride[i], 
                                      padding=padding, 
                                      output_padding=output_padding,
                                      bias=self.use_bias)
      self.add_module("conv_transpose_{}".format(i+1), layer['conv_transpose'])
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      kaiming_init(layer['conv_transpose'], F.relu, self.stride[i], mode='fan_out', use_bias=self.use_bias)
      self.layers.append(layer)
      
  def forward(self, x):
    y = x
    for i, layer in enumerate(self.layers):
      y = layer['conv_transpose'](y)
      if layer['activation'] is not None:
        y = layer['activation'](y)
    return y
  
  

  

  

  

  
  
  

  
  
  
  

  

  
 
    
  
  