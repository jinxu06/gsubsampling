import os
import sys
import numpy as np
from absl import logging
from omegaconf import DictConfig, OmegaConf
import torch
import hydra
import cv2
import pytorch_lightning as pl

from manager import TaskManager

torch.backends.cudnn.allow_tf32 = True

@hydra.main(config_path="conf", config_name="config")
def app(cfg):
  
  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)
  
  # random seeds
  torch.manual_seed(cfg.run.random_seed)
  np.random.seed(cfg.run.random_seed)
  # logging
  if cfg.run.mode == 'train':
    logging.set_verbosity(logging.DEBUG)
    logging.info(cfg.run.exp_name)
  else:
    logging.set_verbosity(logging.INFO)
    
  # torch.backends.cudnn.enabled = False
  # torch.backends.cudnn.allow_tf32 = True
  
  # running mode 
  task_manager = TaskManager(cfg)
  if cfg.run.mode == 'train':
    task_manager.run_training()
  elif cfg.run.mode == 'eval':
    task_manager.run_evaluation()
  elif cfg.run.mode == 'reconstruct':
    task_manager.run_reconstruction()
  elif cfg.run.mode == 'sample':
    task_manager.run_sampling()
  else:
    raise Exception("unknow running mode {}".format(cfg.run.mode))
  
if __name__ == "__main__":
    app()