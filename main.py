"""
Train a neural network model on QM9 dataset.


"""

import os
import argparse

import tensorflow as tf
import numpy as np

# Local modules
from utils.logger import log
from model.mpnn import MPNN
#from dataset import xyz_parser

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--datapath', default='./dataset/qm9.bin', help='Path to qm9.bin')
parser.add_argument('--batch-size', default=100, help='Input batch size for training (default 100)')


def main():
  args = parser.parse_args()

  datapath = args.datapath

  # Create data loading stream
  dataset = tf.data.TFRecordDataset('./dataset/qm9.tfrecords')
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(1)
  iterator = dataset.make_initializable_iterator()
  next_elet = iterator.get_next()


  # Create model for training
  hparams = MPNN.default_hparams()
  model = MPNN(hparams)



  with tf.Session() as sess:

    sess.run(iterator.initializer)
    g, h, e, l = sess.run(next_elet)


    # TODO: need to populate this framework
    for step in range(1, N_step):
      try:
        sess.run(train_op)
      except tf.error.OutOfRangeError:
        # Reload the iterator when it reaches the end of the dataset
        sess.run(iterator.initializer, 
                 feed_dict={_data: #dataset
                            _label: #label
                 })
        sess.run(train_op)
      
      if step % display_step == 0 or step == 1:

        loss, acc = sess.run([loss_op, accuracy])
        log.info()






def _parse_function(record):
  features = {'l': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'g': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'h': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'e': tf.FixedLenFeature((), dtype=tf.string, default_value="")}
  parsed = tf.parse_single_example(record, features)
  l = tf.decode_raw(parsed['l'], tf.float64)
  g = tf.decode_raw(parsed['g'], tf.float64)
  h = tf.decode_raw(parsed['h'], tf.float64)
  e = tf.decode_raw(parsed['e'], tf.float64)

  # TODO: add zero padding for data batching

  #g = tf.reshape(g, )
  #h = tf.reshape(h, [19, 13])

  return g, h, e, l


if __name__ == '__main__':
  main()
