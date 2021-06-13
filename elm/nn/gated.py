import torch 
import torch.nn.functional as F
from .conv_nn import BaseConvNN
from sylvester.layers import GatedConv2d, GatedConvTranspose2d

class GatedConvNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None, 
               use_bias=False,
               h_norm=None,
               g_norm=None):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias)
    device = torch.cuda.current_device()
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      in_channels = self.in_channels if i == 0 else self.out_channels[i-1]
      layer['conv'] = torch.nn.Conv2d(in_channels,
                                      self.out_channels[i]*2, 
                                      kernel_size=self.kernel_size[i], 
                                      stride=self.stride[i], 
                                      bias=self.use_bias)
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      #xavier_uniform_init(layer['conv'].weight, layer['activation'])
      torch.nn.init.constant_(layer['conv'].bias, 0.)
      self.add_module("conv_{}".format(i+1), layer['conv'])
      
      if i < self.num_layers - 1:
        # - Hiddens
        if h_norm == 'in':
          layer['h_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif h_norm == 'bn':
          layer['h_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'h_norm' in layer:
          self.add_module("h_norm_{}".format(i+1), layer['h_norm'])
        # - Gates
        if g_norm == 'in':
          layer['g_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif g_norm == 'bn':
          layer['g_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'g_norm' in layer:
          self.add_module("g_norm_{}".format(i+1), layer['g_norm'])
      
      self.layers.append(layer)
      

  def forward(self, x, ps=None):
    y = x
    for i, layer in enumerate(self.layers):
      _, _, h, w = y.size()
      if i < self.num_layers - 1:
        pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=self.stride[i], compressed=True)
        y = F.pad(y, pad, mode=self.padding_mode)
      y = layer['conv'](y)
      h, g = torch.chunk(y, chunks=2, dim=1)
      if 'h_norm' in layer:
        h = layer['h_norm'](h)
      if layer['activation'] is not None:
        h = layer['activation'](h)
      if 'g_norm' in layer:
        g = layer['g_norm'](g)
      g = F.sigmoid(g)
      y = h * g 
    
    return y#  , None
  
  
class GatedConvTransposeNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='constant',
               activation=F.relu,
               out_activation=None, 
               use_bias=False,
               h_norm=None,
               g_norm=None):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias)
    device = torch.cuda.current_device()
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      in_channels = self.in_channels if i == 0 else self.out_channels[i-1]
      if i == 0:
        layer['conv_transpose'] = torch.nn.ConvTranspose2d(in_channels,
                                        self.out_channels[i]*2, 
                                        kernel_size=self.kernel_size[i], 
                                        stride=self.stride[i], 
                                        output_padding=self.stride[i]-1, 
                                        bias=self.use_bias)                                     
      else:
        layer['conv_transpose'] = torch.nn.ConvTranspose2d(in_channels,
                                        self.out_channels[i]*2, 
                                        kernel_size=self.kernel_size[i], 
                                        stride=self.stride[i], 
                                        padding=(self.kernel_size[i]-1)//2, 
                                        output_padding=self.stride[i]-1, 
                                        bias=self.use_bias)
      # assert self.kernel_size[i] % 2 == 1, "Currently, paddings are properly handled only for odd number kernel size."
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      #xavier_uniform_init(layer['conv_transpose'].weight, layer['activation'])
      torch.nn.init.constant_(layer['conv_transpose'].bias, 0.)
      self.add_module("conv_transpose_{}".format(i+1), layer['conv_transpose'])
      
      if i > 0:
        # - Hiddens
        if h_norm == 'in':
          layer['h_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif h_norm == 'bn':
          layer['h_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'h_norm' in layer:
          self.add_module("h_norm_{}".format(i+1), layer['h_norm'])
        # - Gates
        if g_norm == 'in':
          layer['g_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif g_norm == 'bn':
          layer['g_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'g_norm' in layer:
          self.add_module("g_norm_{}".format(i+1), layer['g_norm'])
        
      self.layers.append(layer)
      

  def forward(self, x, ps=None):
    y = x
    for _, layer in enumerate(self.layers):
      y = layer['conv_transpose'](y)
      h, g = torch.chunk(y, chunks=2, dim=1)
      if 'h_norm' in layer:
        h = layer['h_norm'](h)
      if layer['activation'] is not None:
        h = layer['activation'](h)
      if 'g_norm' in layer:
        g = layer['g_norm'](g)
      g = F.sigmoid(g)
      y = h * g 
    return y
  
  
class GatedEquivariantConvNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='circular',
               activation=F.relu,
               out_activation=None,
               use_bias=False,
               h_norm=None,
               g_norm=None):
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias)
    
    device = torch.cuda.current_device()
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      in_channels = self.in_channels if i == 0 else self.out_channels[i-1] 
      layer['conv'] = torch.nn.Conv2d(in_channels,
                                      self.out_channels[i]*2, 
                                      kernel_size=self.kernel_size[i], 
                                      stride=1, 
                                      bias=self.use_bias)
      if self.stride[i] > 1:
        subsampling = EquivariantSubSampling(scale_ratio=stride[i])
        layer['subsampling'] = subsampling
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      xavier_uniform_init(layer['conv'].weight, layer['activation'])
      torch.nn.init.constant_(layer['conv'].bias, 0.)
      self.add_module("conv_{}".format(i+1), layer['conv'])
      
      if i < self.num_layers - 1:
        # - Hiddens
        if h_norm == 'in':
          layer['h_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif h_norm == 'bn':
          layer['h_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'h_norm' in layer:
          self.add_module("h_norm_{}".format(i+1), layer['h_norm'])
        # - Gates
        if g_norm == 'in':
          layer['g_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif g_norm == 'bn':
          layer['g_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'g_norm' in layer:
          self.add_module("g_norm_{}".format(i+1), layer['g_norm'])
      
      self.layers.append(layer)
  
      
  def forward(self, x, ps=None):
    y = x  
    output_ps = []
    ps_idx = 0
    for i, layer in enumerate(self.layers):
      _, _, h, w = y.size()
      pad = get_same_pad(size=[h, w], kernel_size=self.kernel_size[i], stride=1, compressed=True)
      y = F.pad(y, pad, mode=self.padding_mode)
      y = layer['conv'](y)
      if self.stride[i] > 1:
        if ps is not None:
          y, p = layer['subsampling'](y, ps[:, ps_idx])
        else:
          y, p = layer['subsampling'](y)
        output_ps.append(p)
        ps_idx += 1
      h, g = torch.chunk(y, chunks=2, dim=1)
      if 'h_norm' in layer:
        h = layer['h_norm'](h)
      if layer['activation'] is not None:
        h = layer['activation'](h)
      if 'g_norm' in layer:
        g = layer['g_norm'](g)
      g = F.sigmoid(g)
      y = h * g 
  
    return y, torch.stack(output_ps, 1)
    

class GatedEquivariantConvTransposeNN(BaseConvNN):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               kernel_size,
               stride=1,
               padding_mode='circular',
               activation=F.relu,
               out_activation=None,
               use_bias=False,
               h_norm=None,
               g_norm=None):
    
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding_mode=padding_mode,
                     activation=activation,
                     out_activation=out_activation,
                     use_bias=use_bias)
    
    device = torch.cuda.current_device()
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      in_channels = self.in_channels if i == 0 else self.out_channels[i-1] 
      if self.stride[i] > 1:
        upsampling = EquivariantUpSampling(scale_ratio=self.stride[i])
        layer['upsampling'] = upsampling
      layer['conv_transpose'] = torch.nn.ConvTranspose2d(in_channels,
                                      self.out_channels[i]*2, 
                                      kernel_size=self.kernel_size[i], 
                                      stride=1, 
                                      padding=self.kernel_size[i]-1, 
                                      bias=self.use_bias)
      assert self.kernel_size[i] % 2 == 1, "Currently, paddings are properly handled only for odd number kernel size."
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      xavier_uniform_init(layer['conv_transpose'].weight, layer['activation'])
      torch.nn.init.constant_(layer['conv_transpose'].bias, 0.)
      self.add_module("conv_transpose_{}".format(i+1), layer['conv_transpose'])
      if i > 0:
        # - Hiddens
        if h_norm == 'in':
          layer['h_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif h_norm == 'bn':
          layer['h_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'h_norm' in layer:
          self.add_module("h_norm_{}".format(i+1), layer['h_norm'])
        # - Gates
        if g_norm == 'in':
          layer['g_norm'] = torch.nn.InstanceNorm2d(self.out_channels[i], affine=True).to(device)
        elif g_norm == 'bn':
          layer['g_norm'] = torch.nn.BatchNorm2d(self.out_channels[i]).to(device)
        if 'g_norm' in layer:
          self.add_module("g_norm_{}".format(i+1), layer['g_norm'])
      
      self.layers.append(layer)
  
        
  def forward(self, x, ps=None):
    y = x  
    if ps is not None:
      ps = deque(torch.unbind(ps, dim=1))
    for i, layer in enumerate(self.layers):
      if self.stride[i] > 1:
        y = layer['upsampling'](y, ps.pop())
        #y = layer['upsampling'](y)
      y = F.pad(y, pad=[(self.kernel_size[i]-1)//2 for _ in range(4)], mode=self.padding_mode)  
      y = layer['conv_transpose'](y) * self.stride[i]**2 # dim_group, rescale
      
      h, g = torch.chunk(y, chunks=2, dim=1)
      if 'h_norm' in layer:
        h = layer['h_norm'](h)
      if layer['activation'] is not None:
        h = layer['activation'](h)
      if 'g_norm' in layer:
        g = layer['g_norm'](g)
      g = F.sigmoid(g)
      y = h * g
    return y