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
  dataset = dataset.padded_batch(2, padded_shapes=([30, 30], [30, 50]))
  iterator = dataset.make_initializable_iterator()
  next_elet = iterator.get_next()


  # Create model for training
  hparams = MPNN.default_hparams()
  model = MPNN(hparams)



  with tf.Session() as sess:

    sess.run(iterator.initializer)
    g, h = sess.run(next_elet)
    #log.infov(sess.run(tf.shape(h)))
    log.infov(type(h))


#    # TODO: need to populate this framework
#    for step in range(1, N_step):
#      try:
#        sess.run(train_op)
#      except tf.error.OutOfRangeError:
#        # Reload the iterator when it reaches the end of the dataset
#        sess.run(iterator.initializer, 
#                 feed_dict={_data: #dataset
#                            _label: #label
#                 })
#        sess.run(train_op)
#      
#      if step % display_step == 0 or step == 1:
#
#        loss, acc = sess.run([loss_op, accuracy])
#        log.info()






def _parse_function(record):
  features = {'label': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'adjacency': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'node_state': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'edge_state': tf.FixedLenFeature((), dtype=tf.string, default_value=""),
              'num_nodes': tf.FixedLenFeature((), dtype=tf.int64, default_value=0)}
  parsed = tf.parse_single_example(record, features)
  l = tf.decode_raw(parsed['label'], tf.float64)
  g = tf.decode_raw(parsed['adjacency'], tf.float64)
  h = tf.decode_raw(parsed['node_state'], tf.float64)
  e = tf.decode_raw(parsed['edge_state'], tf.float64)
  num_nodes = tf.cast(parsed['num_nodes'], tf.int32)

  # TODO: add zero padding for data batching
  g = tf.reshape(g, shape=[num_nodes, num_nodes])
  h = tf.reshape(h, shape=[num_nodes, -1])
  l = tf.reshape(l, shape=[12])
  #g = tf.Print(g, [tf.shape(g)], 'awleifyuhaskdfshfksahfuia')

  return g, h


if __name__ == '__main__':
  main()
