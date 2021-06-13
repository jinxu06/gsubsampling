import os
import json
import pickle
import shutil
import numpy as np
from absl import logging
from tqdm import tqdm
import hydra
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
import wandb

from elm.data_loader import DSpritesDataModule, MultiDSpritesDataModule, FashionMNISTDataModule, ClevrDataModule
from elm.model import ConvAE, EquivariantAE, GConvAE, ConvVAE, GConvVAE, EquivariantVAE, MONet, EquivariantMONet
from elm.utils import visualize_images, plot_heatmap_2d, plot_polar_barchart
  

class TaskManager(object):
  
  def __init__(self, config):
    self.config = config 
    if self.config.train.turn_on_profiler:
      self.profiler = pl.profiler.AdvancedProfiler(output_filename="profile_report")
    else:
      self.profiler = pl.profiler.PassThroughProfiler()
    
    if "fiber_group" in self.config.model:
      self.model_name = self.config.model.name + "." + self.config.model.fiber_group
    else:
      self.model_name = self.config.model.name
    self.exp_name = "[{0}]-[model:{1}]-[data:{2}]-[train_size:{3}]-[seed:{4}]".format(self.config.run.exp_name, 
                                                                                      self.model_name, 
                                                                                      self.config.data.name, 
                                                                                      self.config.data.train_set_size, 
                                                                                      self.config.run.random_seed)
    self._load_data()
    self._create_model()
    overwrite = self.config.run.mode == 'train' and (not self.config.run.restore)
    logger, checkpoint_callbacks = self._setup_logging_and_checkpointing(overwrite=overwrite)
    
    if not config.run.use_prog_bar:
      progress_bar_refresh_rate = 0
    else:
      progress_bar_refresh_rate = 1
        
    gpus = 1 if torch.cuda.device_count() >=1 else None
    self.trainer = pl.Trainer(gpus=gpus, 
                              logger=logger, 
                              reload_dataloaders_every_epoch=True, 
                              checkpoint_callback=True, 
                              callbacks=checkpoint_callbacks,
                              profiler=self.profiler,
                              precision=self.config.train.precision, 
                              max_epochs=self.config.run.max_epochs, 
                              progress_bar_refresh_rate=progress_bar_refresh_rate, 
                              num_sanity_val_steps=5,
                              check_val_every_n_epoch=1,
                              limit_val_batches=100)
    
    if self.config.run.restore:
      self._restore_model()
    
    self.results_dir = self.exp_name
    if not os.path.exists(self.results_dir):
      os.makedirs(self.results_dir)
      
  def _setup_logging_and_checkpointing(self, overwrite):
    dir_path = os.path.join(self.config.run.logdir, self.exp_name)
    if overwrite:
      if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
      os.makedirs(dir_path)
      os.makedirs(os.path.join(dir_path, "checkpoints"))
      os.makedirs(os.path.join(dir_path, "wandb_logs"))
    
    
    if 'debug' in self.config.run.exp_name or self.config.run.mode != 'train':
      logger = False
    else:
      if self.config.run.logger == "wandb":
        logger = WandbLogger(name=self.exp_name,
                            save_dir=os.path.join(dir_path, "wandb_logs"),
                            prefix='',
                            project='',
                            entity='')
      elif self.config.run.logger == "tensorboard":
        logger = TensorBoardLogger(os.path.join(dir_path, "tb_logs"), 
                                name="", 
                                version=self.config.run.version)
                               
    checkpoint_callbacks = [ModelCheckpoint(
        # monitor='val_loss_epoch',
        save_last=True,
        dirpath=os.path.join(dir_path, "checkpoints"),
        filename=self.exp_name
    )]
    
    class CustomCallback(Callback):
      
      def on_save_checkpoint(self, trainer, pl_module):
          pl_module.train(False)
          
    if self.config.model.name in ['gconv_ae', 'eqv_ae', 'gconv_vae', 'eqv_vae', 'eqv_mae', 'eqv_monet']:
      checkpoint_callbacks.append(CustomCallback())

    return logger, checkpoint_callbacks
      
  
  def _load_data(self):
    if self.config.data.name == 'dsprites':
      self.data_module = DSpritesDataModule(img_size=self.config.data.img_size,
                                            train_set_size=self.config.data.train_set_size,
                                            val_set_size=self.config.data.val_set_size,
                                            test_set_size=self.config.data.test_set_size,
                                            train_batch_size=self.config.data.batch_size,
                                            constrained_transform=self.config.data.constrained_transform,
                                            n_colors=self.config.data.n_colors,
                                            use_cache=self.config.data.use_cache,
                                            use_bg=False,
                                            datadir=self.config.data.datadir,
                                            random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'multi_dsprites':
      self.data_module = MultiDSpritesDataModule(img_size=self.config.data.img_size,
                                                train_set_size=self.config.data.train_set_size,
                                                val_set_size=self.config.data.val_set_size,
                                                test_set_size=self.config.data.test_set_size,
                                                train_batch_size=self.config.data.batch_size, 
                                                num_workers=self.config.train.num_data_workers, 
                                                datadir=self.config.data.datadir,
                                                random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'clevr6':
      self.data_module = ClevrDataModule(img_size=self.config.data.img_size,
                                          train_set_size=self.config.data.train_set_size,
                                          val_set_size=self.config.data.val_set_size,
                                          test_set_size=self.config.data.test_set_size,
                                          train_batch_size=self.config.data.batch_size, 
                                          under6=True,
                                          num_workers=self.config.train.num_data_workers, 
                                          datadir=self.config.data.datadir,
                                          random_seed=self.config.run.random_seed)
    elif self.config.data.name == 'fashion_mnist':
      self.data_module = FashionMNISTDataModule(img_size=self.config.data.img_size,
                                                train_set_size=self.config.data.train_set_size,
                                                val_set_size=self.config.data.val_set_size,
                                                test_set_size=self.config.data.test_set_size,
                                                train_batch_size=self.config.data.batch_size,
                                                n_rot=self.config.data.n_rot,
                                                transformed=self.config.data.transformed,
                                                constrained_transform=self.config.data.constrained_transform,
                                                colored=self.config.data.colored,
                                                datadir=self.config.data.datadir,
                                                num_workers=self.config.train.num_data_workers,
                                                random_seed=self.config.run.random_seed)
    else:
      raise Exception("Unknow dataset {}".format(self.config.data.name))
  
  def _create_model(self):
    if self.config.model.name == 'conv_ae':
      self.model = ConvAE(in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'eqv_ae':
      self.model = EquivariantAE(in_channels=self.config.data.img_channels, 
                                 out_channels=self.config.data.img_channels,
                                 n_channels=self.config.model.n_channels,
                                 img_size=self.config.data.img_size,
                                 dim_latent=self.config.model.dim_latent,
                                 fiber_group=self.config.model.fiber_group,
                                 n_rot=self.config.model.n_rot,
                                 optim_lr=self.config.train.learning_rate,
                                 profiler=self.profiler)
    elif self.config.model.name == 'gconv_ae':
      self.model = GConvAE(in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          fiber_group=self.config.model.fiber_group,
                          n_rot=self.config.model.n_rot,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'conv_vae':
      self.model = ConvVAE(in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'gconv_vae':
      self.model = GConvVAE(in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          fiber_group=self.config.model.fiber_group,
                          n_rot=self.config.model.n_rot,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'eqv_vae':
      self.model = EquivariantVAE(in_channels=self.config.data.img_channels, 
                                 out_channels=self.config.data.img_channels,
                                 n_channels=self.config.model.n_channels,
                                 img_size=self.config.data.img_size,
                                 dim_latent=self.config.model.dim_latent,
                                 fiber_group=self.config.model.fiber_group,
                                 n_rot=self.config.model.n_rot,
                                 optim_lr=self.config.train.learning_rate,
                                 profiler=self.profiler)
    elif self.config.model.name == 'monet':
      self.model = MONet(in_channels=self.config.data.img_channels, 
                         out_channels=self.config.data.img_channels,
                         n_channels=self.config.model.n_channels,
                         img_size=self.config.data.img_size,
                         dim_latent=self.config.model.dim_latent,
                         K_steps=self.config.model.num_mixtures,
                         kl_l_beta=self.config.model.kl_l_beta,
                         pixel_std_fg=self.config.model.pixel_std_fg,
                         pixel_std_bg=self.config.model.pixel_std_bg,
                         optimizer=self.config.model.optimizer)
    elif self.config.model.name == 'eqv_monet':
      self.model = EquivariantMONet(in_channels=self.config.data.img_channels, 
                         out_channels=self.config.data.img_channels,
                         n_channels=self.config.model.n_channels,
                         img_size=self.config.data.img_size,
                         dim_latent=self.config.model.dim_latent,
                         K_steps=self.config.model.num_mixtures,
                         kl_l_beta=self.config.model.kl_l_beta,
                         pixel_std_fg=self.config.model.pixel_std_fg,
                         pixel_std_bg=self.config.model.pixel_std_bg,
                         optimizer=self.config.model.optimizer,
                         fiber_group=self.config.model.fiber_group,
                         n_rot=self.config.model.n_rot,
                         avg_pool_size=self.config.model.avg_pool_size)
    else:
      raise Exception("Unknow model {}".format(self.config.model.name))
  
  def run_training(self):
    self.trainer.fit(self.model, self.data_module)
  
  def run_evaluation(self):
    self._restore_model()
    self.data_module.setup()
    if self.config.eval.which_set == 'val':
      dataloader = self.data_module.val_dataloader()
      num_examples = len(self.data_module.val_set)
    elif self.config.eval.which_set == 'test':
      dataloader = self.data_module.test_dataloader()
      num_examples = len(self.data_module.test_set)
      
    results = self.trainer.test(self.model, test_dataloaders=dataloader, verbose=False)
    logging.info("-------------------------------------")
    logging.info("EXP Name: " + self.exp_name)
    logging.info("Evaluation on {0} set, {1} examples".format(self.config.eval.which_set, num_examples))
    logging.info(str(results))
    
    with open(os.path.join(self.results_dir, self.exp_name+"-eval.json"), 'w') as fp:
      json.dump(results[0], fp)
      
    return results
      
  
  def run_reconstruction(self):
    self._restore_model()
    self.data_module.setup()
    if self.config.eval.which_set == 'val':
      dataloader = self.data_module.val_dataloader()
    elif self.config.eval.which_set == 'test':
      dataloader = self.data_module.test_dataloader()
    if self.config.model.name in ['eqv_mae', 'monet', 'eqv_monet']:
      inputs, segs = next(iter(dataloader))
      recons, attr = self.model.reconstruct(inputs, segs)
      self.model.visualize(inputs, attr, dirpath=self.results_dir)
    elif self.config.model.name == 'genesis':
      inputs, segs = next(iter(dataloader))
      recons, attr, _, _ = self.model.reconstruct(inputs, segs)
      self.model.visualize(inputs, attr)
    else:
      inputs, _ = next(iter(dataloader))
      recons = self.model.reconstruct(inputs)
      if self.config.data.name == 'shape2d' and self.config.data.num_frames > 1:
        inputs = torch.chunk(inputs, chunks=self.config.data.num_frames, dim=1)
        recons = torch.chunk(recons, chunks=self.config.data.num_frames, dim=1)
        for i in range(self.config.data.num_frames):
          visualize_images(inputs[i], filename=os.path.join(self.results_dir, "inputs_{}.png".format(i+1)))
          visualize_images(recons[i], filename=os.path.join(self.results_dir, "reconstructions_{}.png".format(i+1)))
      else:
        visualize_images(inputs, filename=os.path.join(self.results_dir, "inputs.png"))
        visualize_images(recons, filename=os.path.join(self.results_dir, "reconstructions.png")) 
    logging.info("-------------------------------------")
    logging.info("EXP Name: " + self.exp_name)
    logging.info("Generated inputs and reconstructions")
  
  def run_sampling(self):
    # 
    self._restore_model()
    self.data_module.setup()
    samples = self.model.generate(n_samples=self.config.data.batch_size)
    visualize_images(samples, filename=os.path.join(self.results_dir, "generations.png"))
    logging.info("-------------------------------------")
    logging.info("EXP Name: " + self.exp_name)
    logging.info("Generated samples") 
    
  
  def _restore_model(self):
    if self.config.model.name == 'conv_ae':
      self.model = ConvAE.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                          in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'eqv_ae':
      self.model.train(False)
      self.model = EquivariantAE.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                                 in_channels=self.config.data.img_channels, 
                                 out_channels=self.config.data.img_channels,
                                 n_channels=self.config.model.n_channels,
                                 img_size=self.config.data.img_size,
                                 dim_latent=self.config.model.dim_latent,
                                 fiber_group=self.config.model.fiber_group,
                                 n_rot=self.config.model.n_rot,
                                 optim_lr=self.config.train.learning_rate,
                                 profiler=self.profiler)
    elif self.config.model.name == 'gconv_ae':
      self.model.train(False)
      self.model = GConvAE.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                          in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          fiber_group=self.config.model.fiber_group,
                          n_rot=self.config.model.n_rot,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'conv_vae':
      self.model = ConvVAE.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                          in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'gconv_vae':
      self.model.train(False)
      self.model = GConvVAE.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                          in_channels=self.config.data.img_channels, 
                          out_channels=self.config.data.img_channels,
                          n_channels=self.config.model.n_channels,
                          img_size=self.config.data.img_size,
                          dim_latent=self.config.model.dim_latent,
                          fiber_group=self.config.model.fiber_group,
                          n_rot=self.config.model.n_rot,
                          optim_lr=self.config.train.learning_rate,
                          profiler=self.profiler)
    elif self.config.model.name == 'eqv_vae':
      self.model.train(False)
      self.model = EquivariantVAE.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"), 
                                 in_channels=self.config.data.img_channels, 
                                 out_channels=self.config.data.img_channels,
                                 n_channels=self.config.model.n_channels,
                                 img_size=self.config.data.img_size,
                                 dim_latent=self.config.model.dim_latent,
                                 fiber_group=self.config.model.fiber_group,
                                 n_rot=self.config.model.n_rot,
                                 optim_lr=self.config.train.learning_rate,
                                 profiler=self.profiler)
    elif self.config.model.name == 'monet':
      self.model = MONet.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"),
                         in_channels=self.config.data.img_channels, 
                         out_channels=self.config.data.img_channels,
                         n_channels=self.config.model.n_channels,
                         img_size=self.config.data.img_size,
                         dim_latent=self.config.model.dim_latent,
                         kl_l_beta=self.config.model.kl_l_beta,
                         pixel_std_fg=self.config.model.pixel_std_fg,
                         pixel_std_bg=self.config.model.pixel_std_bg,
                         K_steps=self.config.model.num_mixtures,
                         optimizer=self.config.model.optimizer)
    elif self.config.model.name == 'eqv_monet':
      self.model = EquivariantMONet.load_from_checkpoint(os.path.join(self.config.run.logdir, self.exp_name, "checkpoints/last.ckpt"),
                         in_channels=self.config.data.img_channels, 
                         out_channels=self.config.data.img_channels,
                         n_channels=self.config.model.n_channels,
                         img_size=self.config.data.img_size,
                         dim_latent=self.config.model.dim_latent,
                         kl_l_beta=self.config.model.kl_l_beta,
                         pixel_std_fg=self.config.model.pixel_std_fg,
                         pixel_std_bg=self.config.model.pixel_std_bg,
                         optimizer=self.config.model.optimizer,
                         K_steps=self.config.model.num_mixtures,
                         fiber_group=self.config.model.fiber_group,
                         n_rot=self.config.model.n_rot,
                         avg_pool_size=self.config.model.avg_pool_size)
    else:
      raise Exception("Unknow model {}".format(self.config.model.name))
  
  def _restore_session(self):
    pass 
  
  
