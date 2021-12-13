import numpy as np
import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms.functional as TF
from elm.utils import get_same_pad, get_meshgrid, one_hot

class EquivariantUpSampling(torch.nn.Module):
  
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
    
    """ This implementation of G-Upsampling is currently pragmatic and 
    cannot be easily adapted to work with groups beyond p4m. We will 
    update this code with a new implementation soon during the holiday. 
    So if you plan to use this code after the 2022 new year, remember to 
    git pull for the updated code, which should be easier to use.
    """
    
    b, c, h, w = x.size()
    y = x
    if self.fiber_group == 'trivial':
      s_N = self.scale_factor
    elif self.fiber_group == 'rot_2d':
      s_N, s_H = self.scale_factor
    elif self.fiber_group == 'flip_rot_2d':
      s_N, s_H_N, s_H_H = self.scale_factor
      s_H = s_H_N
      
      
    if self.fiber_group == 'flip_rot_2d' and max(s_H_N, s_H_H) > 1:
      b, c, h, w = y.size()
      c = c // (self.fiber_group_size[0] * self.fiber_group_size[1])
      y = y.view(b, c, self.fiber_group_size[1], self.fiber_group_size[0], h, w)
      
      y_flips = []
      
      for j in range(0, self.n_flip, self.n_flip//(self.fiber_group_size[1]*s_H_H)):
        y_rots = []
        for i in range(0, self.n_rot, self.n_rot//(self.fiber_group_size[0]*s_H_N)):
          if h > 1:
            y_rot = y.repeat(1,1,1,1,4,4)[..., 1:,1:]
            sy = y_rot.size()
            if j % 2 == 1:
              y_rot = TF.vflip(y_rot)
            y_rot = TF.rotate(y_rot.view(sy[0],sy[1]*sy[2]*sy[3],sy[4],sy[5]), angle=i*360./self.n_rot)  
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
    
      index_r = one_hot(p[:, 2:3], depth=s_H).type(torch.int64)[:, 0]
      y = torch.stack([torch.zeros_like(y), y], dim=4)
      index_r = index_r.view(b,1,1,1,s_H,1,1).expand(b,c,self.fiber_group_size[1],self.fiber_group_size[0],s_H,h,w)
      y = torch.gather(y, dim=4, index=index_r, sparse_grad=False).view(b,c,self.fiber_group_size[1],self.fiber_group_size[0]*s_H,h,w)
      
      index_f = one_hot(p[:, 3:], depth=s_H_H).type(torch.int64)[:, 0]
      y = torch.stack([torch.zeros_like(y), y], dim=3)
      index_f = index_f.view(b,1,1,s_H_H,1,1,1).expand(b,c,self.fiber_group_size[1],s_H_H,self.fiber_group_size[0]*s_H,h,w)
      y = torch.gather(y, dim=3, index=index_f, sparse_grad=False).view(b,-1,h,w)
      
    if (self.fiber_group == 'rot_2d' and s_H > 1):
      b, c, h, w = y.size()
      c = c // self.fiber_group_size[0]
      y = y.view(b, c, self.fiber_group_size[0], h, w)
      y_rots = []
      for i in range(0, self.n_rot, self.n_rot//(self.fiber_group_size[0]*s_H)):
        if h > 1:
          y_rot = y.repeat(1,1,1,4,4)[..., 1:,1:]
          sy = y_rot.size()
          y_rot = TF.rotate(y_rot.view(sy[0],sy[1]*sy[2],sy[3],sy[4]), angle=i*360./self.n_rot)  
          y_rot = y_rot.view(sy)
          y_rot = y_rot[..., 2*h-1:3*h-1,2*w-1:3*w-1]
        else:
          y_rot = y
        y_rots.append(y_rot)
      index_rot = p[:, 2:3].view(b, 1, 1, 1, 1).expand_as(y)
      y = torch.stack(y_rots, dim=-1)
      y = torch.gather(y, dim=-1, index=torch.unsqueeze(index_rot, dim=-1), sparse_grad=False)[..., 0]
      
      index_n = one_hot(p[:, 2:], depth=s_H).type(torch.int64)[:, 0]
      y = torch.stack([torch.zeros_like(y), y], dim=3)
      index_n = index_n.view(b,1,1,s_H,1,1).expand(b,c,self.fiber_group_size[0],s_H,h,w)
      y = torch.gather(y, dim=3, index=index_n, sparse_grad=False).view(b,-1,h,w)
    
    if s_N > 1:
      b, c, h, w = y.size()
      p_N = one_hot(p[:, :2], depth=s_N).type(torch.int64)
      index_h, index_w = p_N[:, 0], p_N[:, 1]
      
      y = torch.stack([torch.zeros_like(y), y], dim=3)
      index_h = index_h.view(b,1,1,s_N,1).expand(b,c,h,s_N,w)
      
      y = torch.gather(y, dim=3, index=index_h, sparse_grad=False).view(b,c,h*s_N,w)
      
      y = torch.stack([torch.zeros_like(y), y], dim=4)
      index_w = index_w.view(b,1,1,1,s_N).expand(b,c,h*s_N,w,s_N)
      y = torch.gather(y, dim=4, index=index_w, sparse_grad=False).view(b,c,h*s_N,w*s_N)
      
    return y
  
