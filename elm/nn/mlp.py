import torch 
import torch.nn.functional as F
import pytorch_lightning as pl
from elm.utils import kaiming_init

class MLP(torch.nn.Module):
  
  def __init__(self,
               in_sizes,
               out_sizes,
               activation=F.relu,
               out_activation=None):
    super().__init__()
    assert isinstance(in_sizes, int), "in_sizes should be an instance of int"
    assert isinstance(out_sizes, list), "out_sizes should be an instance of list"
    self.in_sizes = [in_sizes] + out_sizes[:-1]
    self.out_sizes = out_sizes
    self.num_layers = len(out_sizes)
    self.activation = activation 
    self.out_activation = out_activation
    
    self.layers = []
    for i in range(self.num_layers):
      layer = {}
      layer['linear'] = torch.nn.Linear(self.in_sizes[i], self.out_sizes[i])
      self.add_module("linear_{}".format(i+1), layer['linear'])
      layer['activation'] = self.activation if i < self.num_layers - 1 else self.out_activation
      self.layers.append(layer)
    
  def forward(self, x):
    y = x
    for layer in self.layers:
      y = layer['linear'](y)
      if layer['activation'] is not None:
        y = layer['activation'](y)
    return y