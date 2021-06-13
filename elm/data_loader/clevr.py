import os
from sys import getsizeof
import collections
import itertools
from absl import logging
import numpy as np
import torch 
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from .base import DataModule


class ClevrDataModule(DataModule):
  
  def __init__(self, 
               img_size,
               train_set_size,
               val_set_size,
               test_set_size,
               train_batch_size, 
               eval_batch_size=None, 
               under6=False,
               shuffle=False, 
               num_workers=0, 
               datadir="", 
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
    self.under6 = under6

  def setup(self, stage=None):

    images = np.memmap(os.path.join(self.datadir, "clevr-image.npy"), 
                  dtype=np.uint8, mode='r', 
                  shape=(100000, 3, 64, 64))
    masks = np.memmap(os.path.join(self.datadir, "clevr-mask.npy"), 
                  dtype=np.uint8, mode='r', 
                  shape=(100000, 11, 1, 64, 64))
    latents = np.load(os.path.join(self.datadir, "clevr-latent.npz"))
    
    test_mask = np.concatenate([np.zeros(60000), np.ones(images.shape[0]-60000)])
    np.random.RandomState(1).shuffle(test_mask) 
    train_val_mask = np.logical_not(test_mask)

    if self.under6:
        under6_mask = latents['visibility'].sum(1) <= 6
        train_val_mask = np.logical_and(train_val_mask, under6_mask)
        test_mask = np.logical_and(test_mask, under6_mask)
    
    num_train_val_available = int(train_val_mask.sum())
    num_test_available = int(test_mask.sum())
    
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

    self.train_set = ClevrDataset(images, masks, train_indexes, img_size=self.img_size, random_seed=self.random_seed)
    self.val_set = ClevrDataset(images, masks, val_indexes, img_size=self.img_size, random_seed=self.random_seed)
    self.test_set = ClevrDataset(images, masks, test_indexes, img_size=self.img_size, random_seed=self.random_seed)


# Parse masks 
def parse_mask(raw_masks, background_entities=1):
  shape = raw_masks.shape
  masks = np.zeros((1, shape[2], shape[3]), dtype='int')
  # Convert to boolean masks
  cond = np.where(raw_masks[:, 0, :, :] == 255, True, False)
  # Ignore background entities
  num_entities = cond.shape[0]
  for o_idx in range(background_entities, num_entities):
      masks[cond[o_idx:o_idx+1, :, :]] = o_idx + 1
  masks = torch.FloatTensor(masks)
  return masks

    
class ClevrDataset(torch.utils.data.IterableDataset):
  
  def __init__(self, images, masks, indexes, img_size=64, random_seed=123):
    self.images = images
    self.masks = masks
    self.img_size = img_size
    self.rng = np.random.RandomState(random_seed)
    self.indexes = indexes
    
  def __len__(self):
    return self.indexes.shape[0]
    
  def __iter__(self):
    self.rng.shuffle(self.indexes)
    for idx in self.indexes:
        x = self.images[idx]
        x = (torch.FloatTensor(x.copy()) / 255.)
        m = (parse_mask(self.masks[idx]))
        m = m.type(torch.LongTensor)
        yield x, m