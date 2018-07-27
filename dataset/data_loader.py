"""Class to create data pipeline"""

import tensorflow as tf

class DataLoader:
  def __init__(self, data_dir, hparams):
    # TODO: shuffle dataset
    self._create_dataset(data_dir, hparams)

    self.handle = tf.placeholder(tf.string, shape=[])
    self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_iterator.output_types)

  def _create_dataset(self, data_dir, hp):
    """"""
    dataset = tf.data.TFRecordDataset(data_dir)
    dataset = dataset.map(self._parse_function).padded_batch(
      hp.batch_size, 
      padded_shapes=(
        [hp.num_nodes, hp.num_nodes],
        [hp.num_nodes, hp.node_dim],
        [None]))
    train = dataset.take(hp.train_set_num)
    valid = dataset.skip(hp.train_set_num).take(hp.valid_set_num)
    self.train_iterator = train.make_initializable_iterator()
    self.valid_iterator = valid.make_initializable_iterator()

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

