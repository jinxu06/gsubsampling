import os
import collections
import itertools
import numpy as np
import torch 
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from .base import DataModule


class FashionMNISTDataModule(DataModule):
  
  def __init__(self, 
               img_size,
               train_set_size,
               val_set_size,
               test_set_size,
               train_batch_size, 
               eval_batch_size=None, 
               n_rot=1,
               transformed=False,
               constrained_transform="",
               colored=False,
               shuffle=False, 
               num_workers=0, 
               datadir="", 
               random_seed=None):
    
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
    
    self.n_rot = n_rot
    self.transformed = transformed
    self.constrained_transform = constrained_transform
    self.colored = colored

  def setup(self, stage=None):
    
    transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
    ])

    torchvision.datasets.FashionMNIST(root=self.datadir, train=True, download=True)
    X_train, y_train = load_mnist(path=os.path.join(self.datadir, 'FashionMNIST/raw'), kind='train')
    data_set = torch.utils.data.TensorDataset(torch.Tensor(X_train).reshape(-1,1,28,28), torch.Tensor(y_train))

    self.train_set = torch.utils.data.Subset(data_set, range(self.train_set_size))
    self.val_set = torch.utils.data.Subset(data_set, range(40000-self.val_set_size, 40000))
    self.test_set = torch.utils.data.Subset(data_set, range(40000, 40000+self.test_set_size))
    self.train_set.dataset.transform = transform 
    self.val_set.dataset.transform = transform 
    self.test_set.dataset.transform = transform 
    
    if self.transformed:
      max_angle = 90 if 'rotation' in self.constrained_transform else 360
      if self.n_rot > 1:
        rotation = (max_angle / self.n_rot) * torch.randint(low=0, high=self.n_rot, size=(60000, 1))
      elif self.n_rot < 0:
        rotation = torch.randint(low=0, high=max_angle, size=(60000, 1))
      else:
        rotation = float('-inf') * torch.ones(60000, 1)
      if 'flip' in self.constrained_transform:
        flip = torch.zeros(60000, 1)
      else:
        flip = torch.randint(low=0, high=2, size=(60000, 1))
      # scale = torch.randint(low=7, high=15, size=(60000, 1)) * 2
      scale = float('-inf') * torch.ones(60000, 1)
      # scale = torch.randint(low=2, high=5, size=(60000, 1)) * 7
      if 'translation' in self.constrained_transform:
        translation = torch.randint(low=-18, high=0, size=(60000, 2))
      else:
        translation = torch.randint(low=-18, high=19, size=(60000, 2))
    else:
      flip = float('-inf') * torch.ones(60000, 1)
      scale = float('-inf') * torch.ones(60000, 1)
      translation = float('-inf') * torch.ones(60000, 2)
      rotation = float('-inf') * torch.ones(60000, 1)
    if self.colored:
      rgb = torch.rand(60000, 3) * 0.5 + 0.5
    else:
      rgb = float('-inf') * torch.ones(60000, 3)
    transforms = torch.cat([rotation, flip, scale, translation, rgb], dim=1)

    self.train_set = CachedDataset(self.train_set, transforms[:self.train_set_size])
    self.val_set = CachedDataset(self.val_set, transforms[40000-self.val_set_size:40000])
    self.test_set = CachedDataset(self.test_set, transforms[40000:40000+self.test_set_size])

def composed_transformation(x, rotation=0, flip=0, scale=28, translation=(0, 0), rgb=(1., 1., 1.)):
  if rotation > float('-inf'):
    x = TF.rotate(x, int(rotation))
  if flip > 0:
    x = TF.vflip(x)
  if scale > float('-inf'):
    x = TF.resize(x, int(scale))
  padding =  [(64-int(s))//2 for s in x.size()[-2:]]
  x = TF.pad(x, padding=padding)
  t_x, t_y = translation
  if t_x > float('-inf') and t_y > float('-inf'):
    x = torch.roll(x, shifts=(int(t_x), int(t_y)), dims=(-2,-1))
  if min(rgb) > float('-inf'):
    x = torch.cat([x * c for c in rgb], dim=0)
  return x 
    
class CachedDataset(torch.utils.data.IterableDataset):
  
  def __init__(self, dataset, transforms, bg_image=None, use_cache=True):
    self.dataset = dataset 
    self.transforms = transforms
    self.cache = []
    self.use_cache = use_cache
    self.bg_image = bg_image
    
  def __len__(self):
    return len(self.dataset)
      
  def __iter__(self):
    if (not self.use_cache) or len(self.cache) == 0:
      for i, (x, y) in enumerate(self.dataset):
        x = x / 255.
        t = list(self.transforms[i].detach().cpu().numpy())
        x = composed_transformation(x, rotation=t[0], flip=t[1], scale=t[2], translation=t[3:5], rgb=t[5:])
        t = torch.Tensor(t + [y]).type_as(x)
        if self.use_cache:
          self.cache.append((x, t))
        yield (x, t)
    else:
      for data in self.cache:
        yield data



def load_mnist(path, kind='train'):
  import gzip

  """Load MNIST data from `path`"""
  labels_path = os.path.join(path,
                              '%s-labels-idx1-ubyte.gz'
                              % kind)
  images_path = os.path.join(path,
                              '%s-images-idx3-ubyte.gz'
                              % kind)

  with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                            offset=8)

  with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                            offset=16).reshape(len(labels), 784)

  return images, labels
    
  