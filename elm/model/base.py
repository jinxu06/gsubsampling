import os
import sys
from absl import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torch.distributions as tdist 


class AutoEncoderModule(pl.LightningModule, torch.nn.Module):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               n_channels,
               img_size, 
               dim_latent, 
               activation=F.relu,
               readout_fn=None, 
               optim_lr=0.0001, 
               profiler=None):
               
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_channels = n_channels
    self.img_size = img_size
    self.dim_latent = dim_latent
    self.activation = activation
    self.readout_fn = readout_fn 
    self.optim_lr = optim_lr 
    if profiler is None:
      profiler = pl.profiler.PassThroughProfiler()
    self.profiler = profiler
    self.params = None 
    
  def reconstruct(self, x):
    raise NotImplementedError
  
  def compute_loss_and_metrics(self, x, seg=None):
    x_hat = self.reconstruct(x)
    loss = F.mse_loss(x_hat, x)
    logs = {
      "mse": loss
    }
    return loss, logs
  
  def training_step(self, batch, batch_idx):
    x, y = batch 
    with self.profiler.profile("compute loss"):
      if y is not None:
        loss, logs = self.compute_loss_and_metrics(x, y)
      else:
        loss, logs = self.compute_loss_and_metrics(x)
    self.log_dict({f"tr_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return loss 

  def validation_step(self, batch, batch_idx):
    x, y = batch 
    if y is not None:
      loss, logs = self.compute_loss_and_metrics(x, y)
    else:
      loss, logs = self.compute_loss_and_metrics(x)
    self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return loss 

  
  def test_step(self, batch, batch_idx):
    x, y = batch 
    if y is not None:
      loss, logs = self.compute_loss_and_metrics(x, y)
    else:
      loss, logs = self.compute_loss_and_metrics(x)
    self.log_dict({f"test_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    return loss 
  
  def configure_optimizers(self):
    assert self.params is not None, "self.params is None"
    logging.debug("Optim Algo: ADAM")
    return torch.optim.Adam(self.params, lr=self.optim_lr)
  
  def get_progress_bar_dict(self):
    # don't show the version number
    items = super().get_progress_bar_dict()
    items.pop("v_num", None)
    return items
  
  
class VariationalAutoEncoderModule(AutoEncoderModule):
  
  def __init__(self, 
               in_channels, 
               out_channels,
               n_channels,
               img_size, 
               dim_latent, 
               activation=F.relu,
               readout_fn=None, 
               optim_lr=0.0001, 
               profiler=None):
               
    super().__init__(in_channels=in_channels,
                    out_channels=out_channels,
                    n_channels=n_channels,
                    img_size=img_size,
                    dim_latent=dim_latent, 
                    activation=activation,
                    readout_fn=readout_fn,
                    optim_lr=optim_lr, 
                    profiler=profiler)
    
  def reparameterize(self, *params):
    raise NotImplementedError
  
  def generate(self, n_samples):
    raise NotImplementedError
    
  def compute_loss_and_metrics(self, x):
    raise NotImplementedError
  
  def encode(self, x):
    raise NotImplementedError
  
  def decode(self, z):
    raise NotImplementedError
  
  
  

  
    
  
    

