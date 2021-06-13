import numpy as np 
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class GaussianBlur(torch.nn.Module):
  
  def __init__(self, kernel_size, sigma=1.0):
    super().__init__()
    for k in kernel_size:
      assert k % 2==1, "currently only support odd kernel size"
    self.n_dim = len(kernel_size)
    self.kernel_size = kernel_size
    self.sigma = sigma
    inputs = np.zeros(self.kernel_size)
    inputs[tuple([k//2 for k in self.kernel_size])] = np.prod(self.kernel_size)
    kernel = gaussian_filter(inputs, sigma=self.sigma, mode='constant')
    self.register_buffer("kernel", torch.Tensor(kernel))

  def forward(self, x):
    pad = []
    for k in self.kernel_size[::-1]:
      pad += [k//2, k//2]
    x = F.pad(x, pad=pad, mode='circular')
    weights = torch.unsqueeze(torch.unsqueeze(self.kernel.type_as(x), dim=0), dim=0)
    return F.conv3d(x, weights)