import torch
import tensorflow as tf
import os
import numpy as np
from torchvision.utils import make_grid, save_image
import jax.numpy as jnp
from PIL import Image
import PIL


def merge_xy(x,y):
  return jnp.concatenate([x[:,:,:,0:1], y[:,:,:,0:1],x[:,:,:,1:2],y[:,:,:,1:2],x[:,:,:,2:3], y[:,:,:,2:3]], 3)

def image_grid(x):
  size = 128
  channels = 3
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def save_samples(x, r, suf, workdir):
  img = image_grid(x)
  new_name = '%03d' % r
  imgs_dir = 'sr_images'
  tf.io.gfile.makedirs(os.path.join(workdir,imgs_dir))
  name = os.path.join(workdir,imgs_dir,new_name+suf+'.png')
  img = np.clip(img * 255., 0, 255)
  img = np.uint8(img)
  img = Image.fromarray(img)
  #img = img.resize((128, 128))
  img.save(name)

