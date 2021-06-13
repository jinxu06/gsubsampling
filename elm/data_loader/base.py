import numpy as np 
import torch 
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
  
  def __init__(self, 
               img_size,
               train_set_size,
               val_set_size,
               test_set_size,
               train_batch_size, 
               eval_batch_size=None, 
               shuffle=False, 
               num_workers=0, 
               datadir="", 
               random_seed=None):
    super().__init__()
    self.img_size = img_size
    self.train_set_size = train_set_size
    self.val_set_size = val_set_size
    self.test_set_size = test_set_size
    self.train_batch_size = train_batch_size 
    if eval_batch_size is None:
      eval_batch_size = train_batch_size 
    self.eval_batch_size = eval_batch_size
    
    self.shuffle = shuffle    
    self.num_workers = num_workers
    self.datadir = datadir 
    self.random_seed = random_seed
    self.rng = np.random.RandomState(random_seed)
    
    self.train_set = None 
    self.val_set = None 
    self.test_set = None 
    
  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_set, 
                      batch_size=self.train_batch_size,
                      shuffle=self.shuffle, 
                      num_workers=self.num_workers,
                      pin_memory=True)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.val_set, 
                      batch_size=self.eval_batch_size,
                      shuffle=self.shuffle, 
                      num_workers=self.num_workers,
                      pin_memory=True)

  def test_dataloader(self):
    return torch.utils.data.DataLoader(self.test_set, 
                      batch_size=self.eval_batch_size, 
                      shuffle=self.shuffle, 
                      num_workers=self.num_workers,
                      pin_memory=True)