import numpy as np
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
import torch.distributions as tdist
import e2cnn
from elm.utils import get_same_pad, get_meshgrid, one_hot
from .gaussian_blur import GaussianBlur
from elm.utils import unravel_index


class EquivariantFeatureSpace2Group(torch.nn.Module):
  
  def __init__(self, 
               in_channels, 
               fiber_group='trivial', 
               fiber_group_size=(1, 1), 
               conv_kernel_size=5, 
               blur_kernel_size=15, 
               temperature=0.001,
               version=1,
               device='cuda'):
    super().__init__()
    self.in_channels = in_channels
    self.fiber_group = fiber_group
    self.n_rot, self.n_flip = fiber_group_size
    self.conv_kernel_size = conv_kernel_size
    self.blur_kernel_size = blur_kernel_size
    self.temperature = temperature
    self.device = device 
    self.gaussian_blur = GaussianBlur([1, blur_kernel_size, blur_kernel_size], sigma=5.0)
    self.conv = torch.nn.Conv2d(self.in_channels, 1, kernel_size=self.conv_kernel_size, bias=False)
    torch.nn.init.ones_(self.conv.weight)
    self.conv.weight.requires_grad = False
    
  
  def forward(self, x):

    b, c, h, w = x.size()
    c = c // (self.n_rot*self.n_flip)
    x = x.view(b, c, self.n_rot*self.n_flip, h, w)
    x = (x - torch.mean(x, dim=(2,3,4), keepdim=True)).abs()
    x = x.view(b, c, self.n_flip, self.n_rot, h, w)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(b*self.n_flip*self.n_rot, c, h, w)

    pad = get_same_pad(size=[h, w], kernel_size=self.conv_kernel_size, stride=1)
    x = F.pad(x, pad, mode='circular')
    x = self.conv(x) / self.conv_kernel_size ** 2
    x = x.view(b*self.n_flip, 1, self.n_rot, h, w)
    x = self.gaussian_blur(x)
    x = x.view(b, self.n_flip, self.n_rot, h, w) 

    s = x.size()
    p = (x.type(torch.float32)/self.temperature).view(s[0],-1).softmax(dim=-1)
    idx = torch.multinomial(p, 1)
    idx = unravel_index(idx, s[1:])[:, 0]
    return idx.type(torch.long)

class EquivariantSubSampling(torch.nn.Module):
  
  """ This implementation of G-Subsampling is currently pragmatic and 
  cannot be easily adapted to work with groups beyond p4m. We will 
  update this code with a new implementation soon during the holiday. 
  So if you plan to use this code after the 2022 new year, remember to 
  git pull for the updated code, which should be easier to use.
  """
  
  def __init__(self, 
               scale_factor, 
               fiber_group_size=(1,1), 
               fiber_group='trivial', 
               n_rot=1,
               device='cuda'):
    super().__init__()
    self.scale_factor = scale_factor
    self.fiber_group_size = tuple(fiber_group_size)
    self.fiber_group = fiber_group
    self.n_rot = n_rot
    self.device = device 
    self.n_flip = 2 if self.fiber_group == 'flip_rot_2d' else 1

  def forward(self, x, p):
    b, c, h, w = x.size()
    y = x
    if self.fiber_group == 'trivial':
      s_N = self.scale_factor
    elif self.fiber_group == 'rot_2d':
      s_N, s_H = self.scale_factor
    elif self.fiber_group == 'flip_rot_2d':
      s_N, s_H_N, s_H_H = self.scale_factor
      s_H = s_H_N
    
    if s_N > 1:
      
      index_h = p[:, 0:1] + torch.arange(0, h, s_N, dtype=torch.int64, device=x.device).view(1,-1)
      index_h = torch.unsqueeze(torch.unsqueeze(index_h, dim=1), dim=-1)
      index_h = index_h.expand(b,c,h//s_N,w)
      index_w = p[:, 1:2] + torch.arange(0, w, s_N, dtype=torch.int64, device=x.device).view(1,-1)
      index_w = torch.unsqueeze(torch.unsqueeze(index_w, dim=1), dim=-2)
      index_w = index_w.expand(b,c,h//s_N,w//s_N)
      y = torch.gather(y, dim=-2, index=index_h, sparse_grad=False)
      y = torch.gather(y, dim=-1, index=index_w, sparse_grad=False)
      
    if (self.fiber_group == 'rot_2d' and s_H > 1): 
      
      b, c, h, w = y.size()
      c = c // self.fiber_group_size[0]
      y = y.view(b, c, self.fiber_group_size[0], h, w)
      index_rot = p[:, 2:3].view(b, 1, 1, 1, 1).expand_as(y)
      y_rots = []
      for i in range(0, self.n_rot, self.n_rot//self.fiber_group_size[0]):
        if h > 1:
          y_rot = y.repeat(1,1,1,4,4)[..., 1:,1:]
          sy = y_rot.size()
          y_rot = TF.rotate(y_rot.view(sy[0],sy[1]*sy[2],sy[3],sy[4]), angle=-i*360./self.n_rot)  
          y_rot = y_rot.view(sy)
          y_rot = y_rot[..., 2*h-1:3*h-1,2*w-1:3*w-1]
        else:
          y_rot = y
        y_rots.append(y_rot)
      y = torch.stack(y_rots, dim=-1)
      y = torch.gather(y, dim=-1, index=torch.unsqueeze(index_rot, dim=-1), sparse_grad=False)[..., 0]
      
      index_r = p[:, 2:3] + torch.arange(0, self.fiber_group_size[0], s_H, dtype=torch.long, device=x.device).view(1,-1)
      index_r = index_r.view(b, 1, self.fiber_group_size[0]//s_H, 1, 1) 
      index_r = index_r.expand(b, c, self.fiber_group_size[0]//s_H, h, w)
      y = torch.gather(y, dim=-3, index=index_r, sparse_grad=False)
      y = y.view(b, c*self.fiber_group_size[0]//s_H, h, w)
      
    if self.fiber_group == 'flip_rot_2d' and max(s_H_N, s_H_H) > 1:
      
      b, c, h, w = y.size()
      c = c // (self.fiber_group_size[0] * self.fiber_group_size[1])
      y = y.view(b, c, self.fiber_group_size[1], self.fiber_group_size[0], h, w)
      
      y_flips = []
      for j in range(0, self.n_flip, self.n_flip//self.fiber_group_size[1]):
        y_rots = []
        for i in range(0, self.n_rot, self.n_rot//self.fiber_group_size[0]):
          if h > 1:
            y_rot = y.repeat(1,1,1,1,4,4)[..., 1:,1:]
            sy = y_rot.size()
            if j % 2 == 1:
              y_rot = TF.vflip(y_rot)
            y_rot = TF.rotate(y_rot.view(sy[0],sy[1]*sy[2]*sy[3],sy[4],sy[5]), angle=-(-1)**j*i*360./self.n_rot)  
            y_rot = y_rot.view(sy)
            y_rot = y_rot[..., 2*h-1:3*h-1,2*w-1:3*w-1]
          else:
            y_rot = y
          y_rots.append(y_rot)
        y_flips.append(torch.stack(y_rots, dim=-1))
      y = torch.stack(y_flips, dim=-1)

      index_rot = p[:, 2:3].view(b, 1, 1, 1, 1, 1, 1, 1).expand_as(y)[..., 0:1, :]
      y = torch.gather(y, dim=-2, index=index_rot, sparse_grad=False)[..., 0, :]
      index_flip = p[:, 3:].view(b, 1, 1, 1, 1, 1, 1).expand_as(y)[..., 0:1]
      y = torch.gather(y, dim=-1, index=index_flip, sparse_grad=False)[..., 0]
      
      index_r = p[:, 2:3] + torch.arange(0, self.fiber_group_size[0], s_H, dtype=torch.long, device=x.device).view(1,-1)
      index_r = index_r.view(b, 1, 1, self.fiber_group_size[0]//s_H, 1, 1) 
      index_r = index_r.expand(b, c, self.fiber_group_size[1], self.fiber_group_size[0]//s_H, h, w)
      y = torch.gather(y, dim=-3, index=index_r, sparse_grad=False)
      
      index_f = p[:, 3:] + torch.arange(0, self.fiber_group_size[1], s_H_H, dtype=torch.long, device=x.device).view(1,-1)
      index_f = index_f.view(b, 1, self.fiber_group_size[1]//s_H_H, 1, 1, 1) 
      index_f = index_f.expand(b, c, self.fiber_group_size[1]//s_H_H, self.fiber_group_size[0]//s_H, h, w)

      y = torch.gather(y, dim=-4, index=index_f, sparse_grad=False)
      y = y.view(b, -1, h, w)
      
    return y
 
