import colorsys
import math
import collections
import numpy as np 
import torch 
import torch.nn.functional as F

  
def get_meshgrid(start, num, step=1.0):
  arrs = []
  for s, n in zip(start, num):
    arr = torch.arange(start=s, end=s+n*step-0.5, step=step, dtype=torch.float32)
    arrs.append(arr)
  arrs = torch.meshgrid(*arrs)
  grid = torch.stack(arrs, dim=0)
  return grid 

# https://github.com/pytorch/pytorch/issues/35674
def unravel_index(indices, shape):
  """Converts flat indices into unraveled coordinates in a target shape.

  This is a `torch` implementation of `numpy.unravel_index`.

  Args:
    indices: A tensor of indices, (*, N).
    shape: The targeted shape, (D,).

  Returns:
    unravel coordinates, (*, N, D).
  """
  shape = torch.tensor(shape)
  indices = indices % shape.prod()  # prevent out-of-bounds indices
  coord = torch.zeros(indices.size() + shape.size(), dtype=int).to(indices.device)
  for i, dim in enumerate(reversed(shape)):
    coord[..., i] = indices % dim
    indices = indices // dim
  return coord.flip(-1)

def one_hot(batch, depth): 
  s = batch.size()
  ones = torch.eye(depth, device=batch.device)
  batch = ones.index_select(0, torch.reshape(batch, [-1]))
  return torch.reshape(batch, s+torch.Size([depth]))

def hsv2rgb(h,s,v):
  h = h / np.pi / 2.
  return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))


def _cal_same_pad(size, kernel_size, stride):
  output_size = math.ceil(size / stride)
  pad = (output_size - 1) * stride + kernel_size - size 
  return pad // 2, pad - pad // 2
  
def get_same_pad(size, kernel_size, stride, compressed=True):
  if isinstance(size, int):
    size = [size]
  assert isinstance(size, collections.Sequence), "size should be int or sequence"
  if isinstance(kernel_size, int):
    kernel_size = [kernel_size for _ in range(len(size))]
  if isinstance(stride, int):
    stride= [stride for _ in range(len(size))]
    
  pads = []
  for i in range(len(size)):
    pad = _cal_same_pad(size[i], kernel_size[i], stride[i])
    if compressed:
      pads += list(pad)
    else:
      pads.append(pad) 
  return pads


def calculate_nonlinearity_gain(nonlinearity):
  if nonlinearity is None:
    gain = 1
  elif nonlinearity == F.relu:
    gain = torch.nn.init.calculate_gain('relu')
  elif nonlinearity == F.elu:
    gain = 1.247
  elif nonlinearity == F.tanh:
    gain = torch.nn.init.calculate_gain('tanh')
  elif nonlinearity == F.sigmoid:
    gain = torch.nn.init.calculate_gain('sigmoid')
  else:
    raise Exception("Unknown nonlinearity {}".format(nonlinearity))
  return gain

def kaiming_init(layer, nonlinearity, stride=1, mode='fan_in', use_bias=True):
  torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu', mode=mode)
  with torch.no_grad():
    layer.weight /= calculate_nonlinearity_gain(F.relu)
    layer.weight *= calculate_nonlinearity_gain(nonlinearity)
    if mode == 'fan_out':
      layer.weight *= stride
  if use_bias:
    torch.nn.init.zeros_(layer.bias)

def xavier_uniform_init(weight, nonlinearity):
  # https://github.com/pytorch/pytorch/issues/24991
  if nonlinearity is None:
    torch.nn.init.xavier_uniform_(weight, gain=1)
  elif nonlinearity == F.relu:
    torch.nn.init.xavier_uniform_(weight, gain=torch.nn.init.calculate_gain('relu'))
  elif nonlinearity == F.elu:
    torch.nn.init.xavier_uniform_(weight, gain=1.247) 
  elif nonlinearity == F.tanh:
    torch.nn.init.xavier_uniform_(weight, gain=torch.nn.init.calculate_gain('tanh'))
  elif nonlinearity == F.sigmoid:
    torch.nn.init.xavier_uniform_(weight, gain=torch.nn.init.calculate_gain('sigmoid'))
  else:
    torch.nn.init.xavier_uniform_(weight, gain=1)