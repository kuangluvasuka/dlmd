"""
Training and evaluation classes for MPNN.

"""

import tensorflow as tf


class BaseTrain:
  def __init__(self, model, data, hparams):
    self.model = model
    self.data = data
    self.hparams = hparams
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)

    with self.graph.as_default():
      self._make_train_step()
      # TODO: add resume option
      self._initialize_model()

  def train(self):
    #TODO: a loop calling run epoch
    with self.graph.as_default():
      pass

  def _initialize_model(self):
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.sess.run(init_op)

  def _make_train_step(self):
    """Create train step with optimizer in the graph."""
    adj_mat, node_state, label = self.data.iterator.get_next()
    pred = self.model.fprop(node_state, adj_mat)
    self.loss_op = tf.losses.mean_squared_error(pred, label)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    self.train_op = optimizer.minimize(self.loss_op)

  def _run_epoch(self):
    """A single training epoch, which loops over the number of mini-batches."""
    raise NotImplementedError('Subclass should implement run_eopch() method.')


class Trainer(BaseTrain):
  def __init__(self, sess, model, data, hparams):
    super(Trainer, self).__init__(sess, model, data, hparams)

  def _debug_options(self):
    pass

  def run_epoch(self):
    if self.debug:
      pass
    else:



  