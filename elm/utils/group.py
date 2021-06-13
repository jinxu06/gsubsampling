import collections
import numpy as np 
import torch 
import torch.nn.functional as F

  
class CyclicGArray(object):
  
  def __init__(self, elems, N, D=1):
    self.N = N
    self.D = D
    self.elems = torch.remainder(elems, self.N)
    
  def __getitem__(self, idx):
    return CyclicGArray(self.elems[:, idx:idx+1], self.N)
    
  def mul(self, m):
    elems = torch.remainder((self.elems + m.elems), self.N)
    return self.__class__(elems, self.N)
  
  def inv(self):
    elems = torch.remainder(-self.elems, self.N)
    return self.__class__(elems, self.N)
  
  def subgroup(self, quotient_order):
    return self.__class__(self.elems//quotient_order, self.N//quotient_order)
  
  def overgroup(self, quotient_order):
    return self.__class__(self.elems*quotient_order, self.N*quotient_order)
  
  def __str__(self):
    return "{0}, N={1}, D={2}\n".format(self.__class__, self.N, self.D) + str(self.elems)
  
  def quotient_subgroup_decomposition(self, quotient_order):
    representives = torch.remainder(self.elems, quotient_order)
    elems = (self.elems - representives) // quotient_order
    return self.__class__(representives, self.N), \
           self.__class__(elems, self.N//quotient_order)
  
  
class SemidirectProductGArray(object):
  
  def __init__(self, garray_N=None, garray_H=None, elems=None, **kwargs):
    if garray_N is not None and garray_H is not None:
      self.garray_N = garray_N
      self.garray_H = garray_H
    else:
      raise NotImplementedError
    self.N = (self.garray_N.N, self.garray_H.N)
    self.D = self.garray_N.D + self.garray_H.D
    
  @property
  def elems(self):
    return torch.cat([self.garray_N.elems, self.garray_H.elems], axis=1)
  
  def __str__(self):
    return "{0}, N={1}, D={2}\n".format(self.__class__, self.N, self.D) + str(self.elems)
  
  def conj(self, garray_N=None, garray_H=None):
    raise NotImplementedError
    
  def mul(self, m):
    p = self.conj(garray_N=m.garray_N, garray_H=self.garray_H)
    n = self.garray_N.mul(p) # n = p.mul(self.garray_N)
    h = self.garray_H.mul(m.garray_H)   # m.garray_H.mul(self.garray_H)
    r = self.__class__(garray_N=n, garray_H=h)
    return r
  
  def inv(self):
    n = self.conj(garray_N=self.garray_N.inv(), garray_H=self.garray_H.inv())
    h = self.garray_H.inv()
    return self.__class__(garray_N=n, garray_H=h)
  
  def quotient_subgroup_decomposition(self, quotient_order_N, quotient_order_H):
    r_N, s_N = self.garray_N.quotient_subgroup_decomposition(quotient_order_N)
    r_H, s_H = self.garray_H.quotient_subgroup_decomposition(quotient_order_H)
    s_N = self.conj(garray_N=s_N.overgroup(quotient_order_N), garray_H=r_H.inv()).subgroup(quotient_order_N)
    return self.__class__(garray_N=r_N, garray_H=r_H),\
           self.__class__(garray_N=s_N, garray_H=s_H)
      
  
class DihedralGArray(SemidirectProductGArray):
  
  def __init__(self, garray_N=None, garray_H=None, elems=None, N=4, NF=2):
    if garray_N is None or garray_H is None:
      garray_N = CyclicGArray(elems[:, :1], N=N)
      garray_H = CyclicGArray(elems[:, 1:], N=NF)
    else:
      garray_N = CyclicGArray(garray_N.elems, N=garray_N.N)
      garray_H = CyclicGArray(garray_H.elems, N=garray_H.N)
    super().__init__(garray_N=garray_N, garray_H=garray_H)
    
  def conj(self, garray_N=None, garray_H=None):
    if garray_N is None:
      garray_N = self.garray_N
    if garray_H is None:
      garray_H = self.garray_H
    elems = garray_N.elems * (1-garray_H.elems) \
          + garray_N.inv().elems * garray_H.elems
    return garray_N.__class__(elems, garray_N.N, garray_N.D)
    
  
class Rot2dOnCyclicGArray(SemidirectProductGArray):
  
  def __init__(self, garray_N, garray_H):
    super().__init__(garray_N, garray_H)
    
  def conj(self, garray_N=None, garray_H=None):
    if garray_N is None:
      garray_N = self.garray_N
    if garray_H is None:
      garray_H = self.garray_H
  
    cos = torch.cos((2*np.pi)/garray_H.N*garray_H.elems)
    sin = torch.sin((2*np.pi)/garray_H.N*garray_H.elems)
    if garray_H.N in [1, 2, 4]:
      cos, sin = cos.round(), sin.round()
    x0, y0 = garray_N[0].elems, garray_N[1].elems
    x = x0 * cos - y0 * sin
    y = x0 * sin + y0 * cos
    return garray_N.__class__(torch.cat([x, y], dim=1).type(torch.long), garray_N.N)
  
  
class FlipRot2dOnCyclicGArray(SemidirectProductGArray):
  
  def __init__(self, garray_N, garray_H):
    super().__init__(garray_N, garray_H)
    
  def conj(self, garray_N=None, garray_H=None):
    if garray_N is None:
      garray_N = self.garray_N
    if garray_H is None:
      garray_H = self.garray_H
    cos = torch.cos((2*np.pi)/garray_H.N[0]*garray_H.garray_N.elems)
    sin = torch.sin((2*np.pi)/garray_H.N[0]*garray_H.garray_N.elems)
    if garray_H.garray_N.N in [1, 2, 4]:
      cos, sin = cos.round(), sin.round()
    x0, y0 = garray_N[0].elems, garray_N[1].elems
    sign = (-1)**garray_H.garray_H.elems
    x = sign * x0 * cos - y0 * sin
    y = sign * x0 * sin + y0 * cos
    return garray_N.__class__(torch.cat([x, y], dim=1).type(torch.long), garray_N.N)
  
  def quotient_subgroup_decomposition(self, quotient_order_N, quotient_order_H_N, quotient_order_H_H):
    r_N, s_N = self.garray_N.quotient_subgroup_decomposition(quotient_order_N)
    r_H, s_H = self.garray_H.quotient_subgroup_decomposition(quotient_order_H_N, quotient_order_H_H)
    s_N = self.conj(garray_N=s_N.overgroup(quotient_order_N), garray_H=r_H.inv()).subgroup(quotient_order_N)
    return self.__class__(garray_N=r_N, garray_H=r_H),\
           self.__class__(garray_N=s_N, garray_H=s_H)
  
  
def recursive_decomposition(garray, quotient_orders, fiber_group='trivial'):
  reprs = []
  if fiber_group == 'trivial':
    quotient_orders_N = quotient_orders[:, 0]
    for q_N in quotient_orders_N:
      r, garray = garray.quotient_subgroup_decomposition(q_N)
      reprs.append(r)
  elif fiber_group == 'rot_2d':
    quotient_orders_N, quotient_orders_H = quotient_orders[:, 0], quotient_orders[:, 1]
    for q_N, q_H in zip(quotient_orders_N, quotient_orders_H):
      r, garray = garray.quotient_subgroup_decomposition(q_N, q_H)
      reprs.append(r)
  elif fiber_group == 'flip_rot_2d':
    quotient_orders_N, quotient_orders_H_N, quotient_orders_H_H = quotient_orders[:, 0], quotient_orders[:, 1], quotient_orders[:, 2]
    for q_N, q_H_N, q_H_H in zip(quotient_orders_N, quotient_orders_H_N, quotient_orders_H_H):
      r, garray = garray.quotient_subgroup_decomposition(q_N, q_H_N, q_H_H)
      reprs.append(r)
  return torch.stack([r.elems for r in reprs], dim=1).type(torch.long)

def product_of_representives(reprs, quotient_orders, fiber_group='trivial'):
  reprs = torch.unbind(reprs, dim=1)
  quotient_orders_N = quotient_orders[:, 0]
  order_N = np.prod(quotient_orders_N)
  w_N = np.cumprod([1] + list(quotient_orders_N[:-1]))
  cum_garray = None
  if fiber_group == 'trivial':
    for i, r in enumerate(reprs):
      garray = CyclicGArray(r * w_N[i], N=order_N, D=2)
      cum_garray = garray if cum_garray is None else cum_garray.mul(garray)
  elif fiber_group == 'rot_2d':
    quotient_orders_N, quotient_orders_H = quotient_orders[:, 0], quotient_orders[:, 1]
    order_H = np.prod(quotient_orders_H)
    w_H = np.cumprod([1] + list(quotient_orders_H[:-1]))
    for i, r in enumerate(reprs):
      garray = Rot2dOnCyclicGArray(garray_N=CyclicGArray(r[:, :2] * w_N[i], N=order_N, D=2),
                         garray_H=CyclicGArray(r[:, 2:] * w_H[i], N=order_H))
      cum_garray = garray if cum_garray is None else cum_garray.mul(garray)
  elif fiber_group == 'flip_rot_2d':
    quotient_orders_N, quotient_orders_H_N, quotient_orders_H_H = quotient_orders[:, 0], quotient_orders[:, 1], quotient_orders[:, 2]
    order_H_N = np.prod(quotient_orders_H_N)
    w_H_N = np.cumprod([1] + list(quotient_orders_H_N[:-1]))
    order_H_H = np.prod(quotient_orders_H_H)
    w_H_H = np.cumprod([1] + list(quotient_orders_H_H[:-1]))
    for i, r in enumerate(reprs):
      garray_H = DihedralGArray(garray_N=CyclicGArray(r[:, 2:3] * w_H_N[i], N=order_H_N),
                     garray_H=CyclicGArray(r[:, 3:] * w_H_H[i], N=order_H_H))
      garray = FlipRot2dOnCyclicGArray(garray_N=CyclicGArray(r[:, :2] * w_N[i], N=order_N, D=2),
                         garray_H=garray_H)
      cum_garray = garray if cum_garray is None else cum_garray.mul(garray)
      
  return cum_garray




