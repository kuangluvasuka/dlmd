"""Class to create data pipeline"""

import tensorflow as tf

class DataLoader:
  def __init__(self, data_dir, hparams):
    self._dataset = tf.data.TFRecordDataset(data_dir)
    # TODO: shuffle dataset
    self._dataset = self._dataset.map(self._parse_function)
    self._dataset = self._dataset.padded_batch(
      hparams.batch_size,
      padded_shapes=(
        [hparams.num_nodes, hparams.num_nodes],
        [hparams.num_nodes, hparams.node_dim],
        [None]))
    self.train = self._dataset.take(1000)
    self.valid = self._dataset.skip(1000).take(100)
    self.train_iter = self.train.make_initializable_iterator()
    self.valid_iter = self.valid.make_initializable_iterator()

    self.handle = tf.placeholder(tf.string, shape=[])
    self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_iter.output_types)

  def _parse_function(self, record):
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
    g = tf.cast(g, tf.int32)
    h = tf.cast(h, tf.float32)
    l = tf.cast(l, tf.float32)
    num_nodes = tf.cast(parsed['num_nodes'], tf.int32)

    g = tf.reshape(g, shape=[num_nodes, num_nodes])
    h = tf.reshape(h, shape=[num_nodes, -1])
    l = tf.reshape(l, shape=[12])
    #g = tf.Print(g, [tf.shape(g)], 'awleifyuhaskdfshfksahfuia')

    return g, h, l

