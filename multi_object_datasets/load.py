import os
import sys
from multi_object_datasets import clevr_with_masks, multi_dsprites
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import urllib.request
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'multi_dsprites', 'Name of the dataset')
flags.DEFINE_string('datadir', '/tmp', 'Root directory for data storage')
flags.DEFINE_integer('batch_size', 50, 'Batch size for iterating the dataset')
flags.DEFINE_integer('num_examples', 100000, 'Total number of datapoint used.')

def main(argv):

    if FLAGS.dataset == 'multi_dsprites':
        tf_records_path = os.path.join(FLAGS.datadir, "multi_dsprites_colored_on_colored.tfrecords")
        if not os.path.exists(tf_records_path):
            url = "https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords"
            print("Downloading from {}".format(url))
            urllib.request.urlretrieve(url, tf_records_path)
        fullpath = os.path.join(FLAGS.datadir, "multi_dsprites")
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
        dataset = multi_dsprites.dataset(tf_records_path, 'colored_on_colored')
    elif FLAGS.dataset == 'clevr':
        tf_records_path = os.path.join(FLAGS.datadir, "clevr_with_masks_train.tfrecords")
        if not os.path.exists(tf_records_path):
            url = "https://storage.googleapis.com/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords"
            print("Downloading from {}".format(url))
            urllib.request.urlretrieve(url, tf_records_path)
        fullpath = os.path.join(FLAGS.datadir, "clevr")
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)
        dataset = clevr_with_masks.dataset(tf_records_path)

    batched_dataset = dataset.batch(FLAGS.batch_size) 
    iterator = tf.compat.v1.data.make_one_shot_iterator(batched_dataset)

    FLAGS.num_examples = 100000
    mmap_image, mmap_mask = None, None
    n = 0

    latents = {}

    for idx, batch in tqdm(enumerate(iterator)):
        if FLAGS.dataset == 'clevr':
            image = batch['image'][:, 29:221, 64:256, :]
            image = tf.image.resize(image, [64, 64], method=tf.image.ResizeMethod.BILINEAR)
            image = tf.transpose(image, perm=[0,3,1,2])
            mask = batch['mask'][:, :, 29:221, 64:256, :]
            s = mask.get_shape().as_list()
            mask = tf.image.resize(tf.reshape(mask, [s[0]*s[1], s[2], s[3], s[4]]), [64, 64], method=tf.image.ResizeMethod.BILINEAR)
            mask = tf.reshape(mask, [s[0], s[1], 64, 64, s[4]])
            mask = tf.transpose(mask, perm=[0,1,4,2,3])
            
        elif FLAGS.dataset == 'multi_dsprites':
            image = tf.transpose(batch['image'], perm=[0,3,1,2])
            mask = tf.transpose(batch['mask'], perm=[0,1,4,2,3])

        if mmap_image is None:
            mmap_image = np.memmap(os.path.join(fullpath, "{}-image.npy".format(FLAGS.dataset)), dtype=np.uint8, mode='w+', shape=tuple([FLAGS.num_examples]+image.get_shape().as_list()[1:]))
        if mmap_mask is None:
            mmap_mask = np.memmap(os.path.join(fullpath, "{}-mask.npy".format(FLAGS.dataset)), dtype=np.uint8, mode='w+', shape=tuple([FLAGS.num_examples]+mask.get_shape().as_list()[1:]))

        b = min(int(image.shape[0]), FLAGS.num_examples-n)
        mmap_image[n:n+b] = image[:b].numpy()
        mmap_image.flush()
        mmap_mask[n:n+b] = mask[:b].numpy()
        mmap_mask.flush()
        n += b
        for k, v in batch.items():
            if k not in ['image', 'mask']:
                if k not in latents:
                    latents[k] = v.numpy()
                else:
                    latents[k] = np.concatenate([latents[k], v.numpy()], axis=0)

        if n >= FLAGS.num_examples:
            break

    np.savez_compressed(os.path.join(fullpath, "{}-latent.npz".format(FLAGS.dataset)), **latents)
    print("{0} dataset containing {1} examples has been generated at {2}".format(FLAGS.dataset, FLAGS.num_examples, fullpath))

if __name__ == '__main__':
  app.run(main)





