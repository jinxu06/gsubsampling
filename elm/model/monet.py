## Code adapted from https://github.com/applied-ai-lab/genesis/blob/master/models/monet_config.py
import os
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

import numpy as np

from genesis.modules.unet import UNet
import genesis.modules.seq_att as seq_att
from genesis.modules.component_vae import ComponentVAE
from genesis.utils.misc import get_kl
from .base import AutoEncoderModule
from absl import logging
from genesis.utils.misc import average_ari


class MONet(AutoEncoderModule):

    def __init__(self, 
                 in_channels,
                 out_channels,
                 n_channels,
                 img_size,
                 dim_latent,
                 activation=torch.nn.ReLU(),
                 K_steps=5,
                 prior_mode='softmax',
                 montecarlo_kl=False,
                 pixel_bound=True,
                 kl_l_beta=0.5,
                 kl_m_beta=0.5,
                 pixel_std_fg=0.1,
                 pixel_std_bg=0.1,
                 optimizer='RMSProp'):
        super(MONet, self).__init__(in_channels=in_channels,
                                    out_channels=out_channels,
                                    n_channels=n_channels,
                                    img_size=img_size,
                                    dim_latent=dim_latent,
                                    activation=activation)
    
        # Configuration
        self.K_steps = K_steps
        self.num_mixtures = K_steps
        self.prior_mode = prior_mode
        self.mckl = montecarlo_kl
        self.pixel_bound = pixel_bound
        self.img_size = img_size 
        self.optimizer = optimizer
        
        self._create_networks()
        
        # Initialise pixel output standard deviations
        std = pixel_std_fg * torch.ones(1, 1, 1, 1, self.K_steps)
        std[0, 0, 0, 0, 0] = pixel_std_bg  # first step
        self.register_buffer('std', std)
        self.kl_l_beta = kl_l_beta
        self.kl_m_beta = kl_m_beta
        
        self.params = self.parameters()
    
        logging.debug("---------       MONet       ---------")
        logging.debug("-------- Trainable Variables ---------")
        for name, p in self.named_parameters():
          logging.debug("{}, {}".format(name, p.size()))
        logging.debug("--------------------------------------")
        
    def _create_networks(self):
      core = UNet(int(np.log2(self.img_size)-1), 32)
      self.att_process = seq_att.SimpleSBP(core)
      # - Component VAE
      self.comp_vae = ComponentVAE(img_size=self.img_size,
                                    nout=4,# -1,
                                    montecarlo_kl=self.mckl,
                                    pixel_bound=self.pixel_bound,
                                    act=self.activation)
      self.comp_vae.pixel_bound = False
      
        
    def configure_optimizers(self):
      assert self.params is not None, "self.params is None"
      if self.optimizer == 'ADAM':
        logging.debug("Optim Algo: ADAM")
        return torch.optim.Adam(self.params, lr=self.optim_lr)
      elif self.optimizer == 'RMSProp':
        logging.debug("Optim Algo: RMSprop")
        return torch.optim.RMSprop(self.params, lr=self.optim_lr)
      else:
        raise Exception("Unknown optimizer {}".format(self.optimizer))
        
    def _compute_ari(self, x, seg, attr):
      # ARI
      ari, _ = average_ari(attr.log_m_k, seg)
      ari_fg, _ = average_ari(attr.log_m_k, seg, True)
      return ari, ari_fg
    
    def _compute_losses(self, x, attr):
      
      # --- Loss terms ---
      losses = AttrDict()
      # -- Reconstruction loss
      losses['err'] = self.x_loss(x, attr.log_m_k, attr.x_r_k, self.std)
      # # -- Attention mask KL
      losses['kl_m'] = self.kl_m_loss(log_m_k=attr.log_m_k, log_m_r_k=attr.log_m_r_k)
      # -- Component KL
      q_z_k = [Normal(m, s) for m, s in 
                zip(attr.mu_k, attr.sigma_k)]
      kl_l_k = get_kl(
          attr.z_k, q_z_k, len(q_z_k)*[Normal(0, 1)], False)
      losses['kl_l_k'] = [kld.sum(1) for kld in kl_l_k]
      
      # Reconstruction error
      err = losses.err.mean(0)
      # KL divergences
      kl_m, kl_l = torch.tensor(0), torch.tensor(0)
      # -- KL stage 1
      if 'kl_m' in losses:
          kl_m = losses.kl_m.mean(0)
      elif 'kl_m_k' in losses:
          kl_m = torch.stack(losses.kl_m_k, dim=1).mean(dim=0).sum()
      # -- KL stage 2
      if 'kl_l' in losses:
          kl_l = losses.kl_l.mean(0)
      elif 'kl_l_k' in losses:
          kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()
          
      return err, kl_m, kl_l 
    
    def reconstruct(self, x, seg=None):
      
      log_m_k, log_s_k, att_stats = self.att_process(x, self.K_steps-1)

      # --- Reconstruct components ---
      x_m_r_k, comp_stats = self.comp_vae(x, log_m_k)
      # Split into appearances and mask prior
      x_r_k = [item[:, :3, :, :] for item in x_m_r_k]
      m_r_logits_k = [item[:, 3:, :, :] for item in x_m_r_k]
      # Apply pixelbound
      if self.pixel_bound:
          x_r_k = [torch.sigmoid(item) for item in x_r_k]

      # --- Reconstruct input image by marginalising (aka summing) ---
      x_r_stack = torch.stack(x_r_k, dim=4)
      m_stack = torch.stack(log_m_k, dim=4).exp()
      recon = (m_stack * x_r_stack).sum(dim=4)
    
      # --- Reconstruct masks ---
      log_m_r_stack = self.get_mask_recon_stack(
          m_r_logits_k, self.prior_mode, log=True)
      log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
      log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
      
      attr = AttrDict()
      attr['log_m_k'] = log_m_k
      attr['log_m_r_k'] = log_m_r_k
      attr['x_r_k'] = x_r_k
      attr['recon'] = recon
      attr['z_k'] = comp_stats.z_k
      attr['mu_k'] = comp_stats.mu_k
      attr['sigma_k'] = comp_stats.sigma_k
      
      return recon, attr
      
    def compute_loss_and_metrics(self, x, seg=None):
      recon, attr = self.reconstruct(x, seg)
      err, kl_m, kl_l = self._compute_losses(x, attr)
      ari, ari_fg = self._compute_ari(x, seg, attr)
      
      loss = err + self.kl_l_beta * kl_l + self.kl_m_beta * kl_m
      
      mse = F.mse_loss(recon, x)
      logs = {
        "mse": mse,
        "err": err,
        "kl_l": kl_l,
        "kl_m": kl_m,
        "ari": ari,
        "ari_fg": ari_fg
      }
      return loss, logs
    
    def _visualize(self, m=None, gm=None, comp_inputs=None, comp_outputs=None, comp_masked_outputs=None, x=None, x_hat=None):
      if x is not None:
        torchvision.utils.save_image(x, "input.png")
      if x_hat is not None:
        torchvision.utils.save_image(x_hat, "recon.png")
      for i in range(self.num_mixtures):
        if m is not None:
          torchvision.utils.save_image(m[:, i], "mask_{}.png".format(i+1))
        if gm is not None:
          torchvision.utils.save_image(gm[:, i], "gmask_{}.png".format(i+1))
        if comp_inputs is not None:
          torchvision.utils.save_image(comp_inputs[:, i], "comp_inputs_{}.png".format(i+1))
        if comp_outputs is not None:
          torchvision.utils.save_image(comp_outputs[:, i], "comp_outputs_{}.png".format(i+1))
        if comp_masked_outputs is not None:
          torchvision.utils.save_image(comp_masked_outputs[:, i], "comp_masked_outputs_{}.png".format(i+1))
          
    def visualize(self, x, attr, dirpath=None):
      if dirpath is None:
        dirpath = "."
      torchvision.utils.save_image(x, os.path.join(dirpath, "input.png"))
      if 'recon' in attr:
        torchvision.utils.save_image(attr.recon, os.path.join(dirpath, "recon.png"))
      for k in range(self.num_mixtures):
        if 'log_m_k' in attr:
          torchvision.utils.save_image(attr.log_m_k[k].exp(), os.path.join(dirpath, "mask_{}.png".format(k+1)))
        if 'mx' in attr:
          torchvision.utils.save_image(attr.mx[k], os.path.join(dirpath, "masked_inputs_{}.png".format(k+1)))
        if 'x_r_k' in attr:
          torchvision.utils.save_image(attr.x_r_k[k], os.path.join(dirpath, "comp_outputs_{}.png".format(k+1)))
    
    @staticmethod
    def x_loss(x, log_m_k, x_r_k, std, pixel_wise=False):
        # 1.) Sum over steps for per pixel & channel (ppc) losses
        p_xr_stack = Normal(torch.stack(x_r_k, dim=4), std)
        log_xr_stack = p_xr_stack.log_prob(x.unsqueeze(4))
        log_m_stack = torch.stack(log_m_k, dim=4)
        log_mx = log_m_stack + log_xr_stack
        err_ppc = -torch.logsumexp(log_mx, dim=4) 
        # 2.) Sum accross channels and spatial dimensions
        if pixel_wise:
            return err_ppc
        else:
            return err_ppc.sum(dim=(1, 2, 3))

    def get_features(self, image_batch):
        with torch.no_grad():
            _, _, _, _, comp_stats = self.forward(image_batch)
            return torch.cat(comp_stats.z_k, dim=1)

    def get_mask_recon_stack(self, m_r_logits_k, prior_mode, log):
        if prior_mode == 'softmax':
            if log:
                return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
            return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        elif prior_mode == 'scope':
            log_m_r_k = []
            log_scope = torch.zeros_like(m_r_logits_k[0])
            for step, logits in enumerate(m_r_logits_k):
                if step == self.K_steps - 1:
                    log_m_r_k.append(log_scope)
                else:
                    log_m = F.logsigmoid(logits)
                    log_neg_m = F.logsigmoid(-logits)
                    log_m_r_k.append(log_scope + log_m)
                    log_scope = log_scope +  log_neg_m
            log_m_r_stack = torch.stack(log_m_r_k, dim=4)
            return log_m_r_stack if log else log_m_r_stack.exp()
        else:
            raise ValueError("No valid prior mode.")

    def kl_m_loss(self, log_m_k, log_m_r_k):
        batch_size = log_m_k[0].size(0)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        # Lower bound to 1e-5 to avoid infinities
        m_stack = torch.max(m_stack, torch.tensor(1e-5).type_as(m_stack))
        m_r_stack = torch.max(m_r_stack, torch.tensor(1e-5).type_as(m_r_stack))
        q_m = Categorical(m_stack.view(-1, self.K_steps))
        p_m = Categorical(m_r_stack.view(-1, self.K_steps))
        kl_m_ppc = kl_divergence(q_m, p_m).view(batch_size, -1)
        return kl_m_ppc.sum(dim=1)

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        # Sample latents
        z_batched = Normal(0, 1).sample((batch_size*K_steps, self.comp_vae.ldim))
        # Pass latent through decoder
        x_hat_batched = self.comp_vae.decode(z_batched)
        # Split into appearances and masks
        x_r_batched = x_hat_batched[:, :3, :, :]
        m_r_logids_batched = x_hat_batched[:, 3:, :, :]
        # Apply pixel bound to appearances
        if self.pixel_bound:
            x_r_batched = torch.sigmoid(x_r_batched)
        # Chunk into K steps
        x_r_k = torch.chunk(x_r_batched, K_steps, dim=0)
        m_r_logits_k = torch.chunk(m_r_logids_batched, K_steps, dim=0)
        # Normalise masks
        m_r_stack = self.get_mask_recon_stack(
            m_r_logits_k, self.prior_mode, log=False)
        # Apply masking and sum to get generated image
        x_r_stack = torch.stack(x_r_k, dim=4)
        gen_image = (m_r_stack * x_r_stack).sum(dim=4)
        # Tracking
        log_m_r_k = [item.squeeze(dim=4) for item in
                     torch.split(m_r_stack.log(), 1, dim=4)]
        stats = AttrDict(gen_image=gen_image, x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x*m.exp() for x, m in zip(x_r_k, log_m_r_k)])
        return gen_image, stats
