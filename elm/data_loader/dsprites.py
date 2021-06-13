import os
from sys import getsizeof
import collections
import itertools
from absl import logging
import numpy as np
import torch 
import torchvision
import pytorch_lightning as pl
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from .base import DataModule
import urllib.request

class DSpritesDataModule(DataModule):
  
  def __init__(self, 
               img_size,
               train_set_size,
               val_set_size,
               test_set_size,
               train_batch_size, 
               eval_batch_size=None, 
               constrained_transform='translation_rotation',
               n_colors=-1,
               shuffle=False, 
               num_workers=0, 
               datadir="", 
               use_cache=True,
               use_bg=False,
               random_seed=123):
    
    super().__init__(img_size=img_size,
                    train_set_size=train_set_size,
                    val_set_size=val_set_size,
                    test_set_size=test_set_size,
                    train_batch_size=train_batch_size, 
                    eval_batch_size=eval_batch_size, 
                    shuffle=shuffle, 
                    num_workers=num_workers, 
                    datadir=datadir, 
                    random_seed=random_seed)
    self.constrained_transform = constrained_transform
    self.n_colors = n_colors
    self.use_cache = use_cache
    self.use_bg = use_bg

  def setup(self, stage=None):
    
    if not os.path.exists(os.path.join(self.datadir, 'images_full.npy')):
      dirpath = self.datadir
      url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
      urllib.request.urlretrieve(url, os.path.join(dirpath, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))
      dataset_zip = np.load(os.path.join(dirpath, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), mmap_mode='r')
      imgs = dataset_zip['imgs'] 
      latents_values = dataset_zip['latents_values']
      latents_classes = dataset_zip['latents_classes']
      
      np.save(os.path.join(dirpath, "images_full.npy"), imgs)
      np.save(os.path.join(dirpath, "latents_values_full.npy"), latents_values)
      np.save(os.path.join(dirpath, "latents_classes_full.npy"), latents_classes)
    
    images = np.load(os.path.join(self.datadir, 'images_full.npy'), mmap_mode='r')
    latents = np.load(os.path.join(self.datadir, 'latents_values_full.npy'), mmap_mode='r')
    test_mask = np.concatenate([np.zeros(400000), np.ones(images.shape[0]-400000)])
    np.random.RandomState(1).shuffle(test_mask) 
    if self.n_colors > 0:
      color_palette = np.random.RandomState(1).rand(self.n_colors, 3) * 0.5 + 0.5 
    else:
      color_palette = None
    # fix this seed even when random_seed are changed so that train/test split will always be the same.
    train_val_mask = np.logical_not(test_mask)
    
    pos_cond_mask, ori_cond_mask = None, None
    if 'translation' in self.constrained_transform:
      pos_cond_mask = np.logical_and(latents[:, -2] <= 0.5, latents[:, -1] <= 0.5)
    if 'rotation' in self.constrained_transform:
      ori_cond_mask = latents[:, -3] <= np.pi/2
    shape_cond_mask = None
    if 'square' in self.constrained_transform:
      shape_cond_mask = latents[:, 1] == 1
    elif 'ellipse' in self.constrained_transform:
      shape_cond_mask = latents[:, 1] == 2
    elif 'heart' in self.constrained_transform:
      shape_cond_mask = latents[:, 1] == 3
      
    for m in [pos_cond_mask, ori_cond_mask, shape_cond_mask]:
      if m is not None:
        train_val_mask = np.logical_and(train_val_mask, m)
        test_mask = np.logical_and(test_mask, m)
    
    num_train_val_available = int(train_val_mask.sum())
    num_test_available = int(test_mask.sum())
    
    logging.info("constraint: {}".format(self.constrained_transform))
    logging.info("train_val available: {0}, test_available: {1}".format(num_train_val_available, num_test_available))
    logging.info("train/val/test: {0}/{1}/{2}".format(self.train_set_size, self.val_set_size, self.test_set_size))
    
    assert self.train_set_size + self.val_set_size <= num_train_val_available, \
      "train_set_size {0} + val_set_size {1} greater than available {2}".format(self.train_set_size, self.val_set_size, num_train_val_available)
    assert self.test_set_size <= num_test_available, \
      "test_set_size {0} greater than available {1}".format(self.test_set_size, num_test_available)
      
    train_val_indexes = np.nonzero(train_val_mask)[0]
    self.rng.shuffle(train_val_indexes)
    train_indexes, val_indexes = train_val_indexes[:self.train_set_size], train_val_indexes[-self.val_set_size:]
    test_indexes = np.nonzero(test_mask)[0]
    self.rng.shuffle(test_indexes)
    test_indexes = test_indexes[:self.test_set_size]
      
    self.train_set = DSpritesDataset(images, latents, train_indexes, color_palette=color_palette, use_cache=self.use_cache, random_seed=self.random_seed)
    self.val_set = DSpritesDataset(images, latents, val_indexes, color_palette=color_palette, use_cache=self.use_cache, random_seed=self.random_seed)
    self.test_set = DSpritesDataset(images, latents, test_indexes, color_palette=color_palette, use_cache=self.use_cache, random_seed=self.random_seed)
  
    
class DSpritesDataset(torch.utils.data.IterableDataset):
  
  def __init__(self, images, latents, indexes, color_palette=None, bg_image=None, use_cache=True, random_seed=123):
    self.images = images
    self.latents = latents
    self.indexes = indexes
    self.cache = []
    self.use_cache = use_cache
    self.bg_image = bg_image
    self.rng = np.random.RandomState(random_seed)
    self.data_processor = DataProcessor(color_palette)
    
  def __len__(self):
    return self.indexes.shape[0]
  
  def __iter__(self):
    if (not self.use_cache) or len(self.cache) == 0:
      for idx in self.indexes:
        x, y = self.images[idx], self.latents[idx]
        x = torch.FloatTensor(x.copy())
        y = torch.FloatTensor(y.copy())
        x, y = self.data_processor(x, y, self.bg_image)
        if self.use_cache:
          self.cache.append((x, y))
        yield x, y
    else:
      for data in self.cache:
        yield data
        
        
class DataProcessor(torch.nn.Module):
  
  def __init__(self, color_palette):
    super().__init__()
    self.color_palette = color_palette
    if self.color_palette is not None:
      self.color_palette= torch.Tensor(self.color_palette)
      self.n_colors = self.color_palette.size()[0]
    self.random_crop = torchvision.transforms.RandomCrop(size=64)
  
  def forward(self, x, y, bg_image=None):
    x= torch.unsqueeze(x, dim=0).repeat(3,1,1)
    if self.color_palette is None:
      colors = torch.rand(3,1,1) * 0.5 + 0.5
    else:
      colors = self.color_palette[torch.randint(low=0, high=self.n_colors, size=())].view(3,1,1)
    x = x * colors
    y = torch.cat([y, colors.view(3)])
    return x, y
  