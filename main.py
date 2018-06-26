"""
Train a neural network model on QM9 dataset.


"""

import os
import argparse

import tensorflow as tf
import numpy as np

# Local modules
from utils.logger import log
#from dataset import xyz_parser

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--datapath', default='./dataset/qm9.bin', help='Path to qm9.bin')
parser.add_argument('--batch-size', default=100, help='Input batch size for training (default 100)')


def main():
  args = parser.parse_args()

  datapath = args.datapath

  dataset = tf.data.TFRecordDataset('./dataset/qm9.tfrecords')
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(1)
  iterator = dataset.make_initializable_iterator()
  next_elet = iterator.get_next()

  with tf.Session() as sess:

    sess.run(iterator.initializer)
    g, h, e, l = sess.run(next_elet)
    #log.infov(e)

def _parse_function(record):
  features = {'l': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'g': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'h': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'e': tf.FixedLenFeature((), dtype=tf.string, default_value="")}

  parsed = tf.parse_single_example(record, features)
  l = tf.decode_raw(parsed['l'], tf.float64)
  g = tf.decode_raw(parsed['g'], tf.float64)
  h = tf.decode_raw(parsed['h'], tf.float64)
  #e = tf.decode_raw(parsed['e'].values, tf.float32)

  # TODO: add zero padding for data batching

  #g = tf.reshape(g, )
  #h = tf.reshape(h, [19, 13])

  return g, h, e, l


if __name__ == '__main__':
  main()
